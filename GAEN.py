import numpy as np
import tensorflow as tf
from utils.process import *
from configs import *
from models import *
from drpgat import DRPGAT
import scipy
from utils.NodeMinibatchIterator import NodeMinibatchIterator
from utils.eval import *
from utils.link_prediction import *

import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

np.random.seed(123)

FLAGS = tf.flags.FLAGS
time_steps = FLAGS.time_steps
lrate_gcn = FLAGS.learning_rate
task = FLAGS.task

print('time_steps=', time_steps)

# load graphs and features
if task == 'link_prediciton':
    graphs, adjs = load_graphs(FLAGS.dataset)
#     graphs, adjs, labels = load_graphs_school(FLAGS.dataset)
    # Load evaluation data.
    train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
    get_evaluation_data(adjs, time_steps, FLAGS.dataset) 
elif task == 'node_classification':
#     graphs, adjs, feats, labels  = load_graphs_classification(FLAGS.dataset)
    graphs, adjs, labels = load_graphs_school(FLAGS.dataset)
    train_nodes = get_evaluation_classification_data(FLAGS.dataset, labels.shape[0], time_steps) 
else:
    raise Exception("choose task between link_prediction and node_classification")


# load graphs and features
# graphs, adjs = load_graphs(FLAGS.dataset)

context_pairs_train = get_context_pairs(graphs, time_steps)
# # Load evaluation data.
# train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
#     get_evaluation_data(adjs, time_steps, FLAGS.dataset)  

# unify the number of nodes across all networks
for i in range(time_steps-1):
    new_G = nx.MultiGraph()
    new_G.add_nodes_from(graphs[time_steps - 1].nodes(data=True))
    for e in graphs[i].edges():
        new_G.add_edge(e[0], e[1])
        
    graphs[i] = new_G
    adjs[i] = nx.adjacency_matrix(new_G)    



# calculate dynamic node patterns for all networks
p_covss = cal_patterns(adjs, time_steps, FLAGS.n_component)
print("p_covss.size:",len(p_covss))
# p_covss =[normalize_pattern(p_covs)[1] for p_covs in p_covss]

# adj_train = [normalize_graph_gcn(adj) for adj in adjs]
adj_train = [normalize_graph_gcn(adj)[1] for adj in adjs]

if FLAGS.featureless:
    feats = [scipy.sparse.identity(adjs[time_steps - 1].shape[0]).tocsr()[range(0, x.shape[0]), :] for x in adjs if
             x.shape[0] <= adjs[time_steps - 1].shape[0]]

n_node = feats[0].shape[0]
num_features = feats[0].shape[1]

print("num_features:", num_features)

feats_train =[preprocess_features(feat)[1] for feat in feats]
num_features_nonzeros = [x[1].shape for x in feats_train]

phs = {
    'node_1': [ tf.compat.v1.placeholder(tf.int64, shape=(None,), name="node_1") for _ in range(time_steps)],
    # [None,1] for each time step.
    'node_2': [ tf.compat.v1.placeholder(tf.int64, shape=(None,), name="node_2") for _ in range(time_steps)],
    # [None,1] for each time step.
    'batch_nodes':  tf.compat.v1.placeholder(tf.int64, shape=(None,), name="batch_nodes"),  # [None,1]
    'feats': [tf.compat.v1.sparse_placeholder(tf.float32, shape=(None, num_features), name="feats") for _ in
                 range(time_steps)],
    'adjs': [tf.compat.v1.sparse_placeholder(tf.float32, shape=(None, None), name="adjs") for i in
             range(time_steps)],
    'p_covss': [ tf.compat.v1.placeholder(tf.float32, shape=(None, None), name="p_covss") for i in
             range(time_steps)],
    'num_features_nonzeros': [ tf.compat.v1.placeholder(tf.int64) for i in range(time_steps)],
    'dropout_prob':  tf.compat.v1.placeholder_with_default(0., shape=())
}

# layer2_embeds = Sp_GCN_LSTM(num_layers=2, 
#             hidden_size=FLAGS.hidden_size, 
#             seq_len=time_steps,
#             input_dim=num_features,
#             hidden_dim=FLAGS.hidden_dim,
#             output_dim=FLAGS.output_dim,
#             act=tf.nn.relu,
#             dropout_prob=phs['dropout_prob'],
#             num_features_nonzeros=phs['num_features_nonzeros']
#             )(phs['adjs'], phs['feats'], sparse=True)

DRPGAT_model = DRPGAT(act=tf.nn.elu,
                      n_node=n_node,
                      input_dim=num_features,
                      output_dim=FLAGS.output_dim,
                      seq_len=time_steps,
                      n_heads=FLAGS.num_heads,
                      attn_drop=phs['dropout_prob'],
                      ffd_drop=phs['dropout_prob'],
                      residual=False,
                      sparse_inputs=True
                      )
layer2_embeds = DRPGAT_model(phs['adjs'], phs['feats'], phs['p_covss'])

minibatchIterator = NodeMinibatchIterator(graphs, feats_train, adj_train,
                                          phs, time_steps, p_covss=p_covss, batch_size=FLAGS.batch_size,
                                          context_pairs=context_pairs_train,
                                          num_features_nonzeros=num_features_nonzeros,
                                          use_pattern=True)

degrees = minibatchIterator.degs

proximity_labels = [tf.expand_dims(tf.cast(phs['node_2'][t], tf.int64), 1)
                    for t in range(len(phs['feats']))]  # [B, 1]

proximity_neg_samples = []
for t in range(len(phs['feats'])-1):
        proximity_neg_samples.append(tf.nn.fixed_unigram_candidate_sampler(
        true_classes=proximity_labels[t],
        num_true=1,
        num_sampled=FLAGS.neg_sample_size,
        unique=False,
        range_max=len(degrees[t]),
        distortion=0.75,
        unigrams=degrees[t].tolist())[0])

def prediction_loss(total_embeds, node_pas, node_pbs, node_nbs):
    graph_loss = tf.constant(0.0)
    for t in range(time_steps-1):
#     for t in range(time_steps-2, time_steps-1):
        out_embeds = total_embeds[t]
        nodes_pa = node_pas[t]
        nodes_pb = node_pbs[t]
        nodes_nb = node_nbs[t]
        
#         if tf.size(nodes_pa) == 0:
#             continue
        
        embeds_pa = tf.nn.embedding_lookup(out_embeds, nodes_pa)
        embeds_pb = tf.nn.embedding_lookup(out_embeds, nodes_pb)
        embeds_nb = tf.nn.embedding_lookup(out_embeds, nodes_nb)
        
        pos_score, neg_score = cal_score(embeds_pa, embeds_pb, embeds_nb)
        pos_ent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(pos_score), logits=pos_score)
        neg_ent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(neg_score), logits=neg_score)
#         graph_loss += tf.reduce_mean(tf.maximum(.0, tf.subtract(0.1, tf.subtract(pos_ent, neg_ent))))
        graph_loss += tf.reduce_mean(pos_ent) + FLAGS.neg_weight * tf.reduce_mean(neg_ent)

    reg_loss= 0
    for var in DRPGAT_model.var.values():
        reg_loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
#     if len([v for v in tf.trainable_variables() if "sp_drgat" in v.name and "bias" not in v.name]) > 0:
#         reg_loss += tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
#                                    if "sp_drgat" in v.name and "bias" not in v.name]) * FLAGS.weight_decay

#     for v in tf.trainable_variables():
#         reg_loss += tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
#                                    if "sp_drgat" in v.name and "bias" not in v.name]) * FLAGS.weight_decay
    graph_loss += reg_loss
    
    return graph_loss, reg_loss

with tf.name_scope('optimizer'):
    
    loss, reg_loss = prediction_loss(total_embeds=layer2_embeds, 
                           node_pas=phs['node_1'], 
                           node_pbs=phs['node_2'],
                           node_nbs=proximity_neg_samples)
    
#     trainable_params = tf.trainable_variables()
#     gradients = tf.gradients(loss, trainable_params)
#     clip_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
    optimizer = tf.train.AdamOptimizer(learning_rate=lrate_gcn)
#     opt_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))
    opt_op = optimizer.minimize(loss) 

# Initialize session
config = tf.ConfigProto()
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

for epoch in range(FLAGS.epochs):
    minibatchIterator.shuffle()
    
    while not minibatchIterator.end():
        
        feed_dict_train = minibatchIterator.next_minibatch_feed_dict()
        feed_dict_train.update({phs['dropout_prob']:FLAGS.dropout_rate})
        _, train_loss, regu_loss, embeds = sess.run((opt_op, loss, reg_loss, layer2_embeds), 
                                    feed_dict=feed_dict_train)
    
    if epoch % 25 == 0:
        print("Epoch:", '%04d' % (epoch + 1),
              "reg_loss=", "{:.5f}".format(regu_loss),
              "train_loss=", "{:.5f}".format(train_loss)
              )
    
#     if (epoch + 1) % FLAGS.test_freq == 0:
feed_dict_train.update({phs['dropout_prob']:0})
emb = sess.run(layer2_embeds, feed_dict=feed_dict_train)[-1]
#         emb = np.array(emb)

if task == 'link_prediciton':
    # Use external classifier to get validation and test results.
    val_results, test_results = evaluate_classifier(train_edges,
                                                          train_edges_false, val_edges, val_edges_false, test_edges,
                                                          test_edges_false, emb, emb)

    epoch_auc_val_micro = val_results[0]
    epoch_auc_val_macro = val_results[1]
    epoch_acc_val = val_results[2]
    
    epoch_auc_test_micro = test_results[0]
    epoch_auc_test_macro = test_results[1]
    epoch_acc_test = test_results[2]
    
    print(
      "val_Acc=", "{:.5f}".format(epoch_acc_val),
      "val_AUC=", "{:.5f}".format(epoch_auc_val_macro),
      "test_Acc=", "{:.5f}".format(epoch_acc_test),
      "test_AUC=", "{:.5f}".format(epoch_auc_test_macro))  
elif task == 'node_classification':
    average_accs = evaluate_node_classification(emb, labels, train_nodes)
    print("node classification acc [train_ratios=0.3, 0.7]:", [average_accs[0],average_accs[2]])       
    