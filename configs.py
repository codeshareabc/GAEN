import tensorflow as tf
 
tf.flags.DEFINE_string('f', '', '')
 
# general parameters 
tf.flags.DEFINE_string('dataset', 'enron', 'Dataset')
tf.flags.DEFINE_string('task', 'link_prediciton', 'link_prediciton or node_classification')
tf.flags.DEFINE_boolean('featureless', True, 'Node features')
tf.flags.DEFINE_integer('time_steps', 10, 'Number of time steps')
tf.flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate for self-attention model.')
tf.flags.DEFINE_float('dropout_rate', 0.5, 'Dropout rate')
tf.flags.DEFINE_float('neg_weight', 2, 'Negative link prediction weight')
tf.flags.DEFINE_float('weight_decay', 5e-3, 'Weight for L2 loss on embedding matri')
tf.flags.DEFINE_integer('hidden_dim', 128, 'Hidden embed size')
tf.flags.DEFINE_integer('hidden_size', 128, 'LSTM hidden size')
tf.flags.DEFINE_integer('output_dim', 128, 'Output embed size')
tf.flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')
tf.flags.DEFINE_integer('neg_sample_size', 2, 'number of negative samples')
tf.flags.DEFINE_integer('n_component', 10, 'n_component')
tf.flags.DEFINE_integer('walk_len', 5, 'Walk len')
tf.flags.DEFINE_integer('test_freq', 40, 'test_freq')
tf.flags.DEFINE_integer('batch_size', 200, 'batch_size')
tf.flags.DEFINE_float('max_gradient_norm', 1.0, 'Clip gradients to this norm')
       
# GAT parameters
tf.flags.DEFINE_integer('num_heads', 8, 'Number of heads in each GAT layer')
#  


# general parameters 
# tf.flags.DEFINE_string('dataset', 'school', 'Dataset')
# tf.flags.DEFINE_string('task', 'node_classification', 'link_prediciton or node_classification')
# tf.flags.DEFINE_boolean('featureless', True, 'Node features')
# tf.flags.DEFINE_integer('time_steps', 40, 'Number of time steps')
# tf.flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate for self-attention model.')
# tf.flags.DEFINE_float('dropout_rate', 0.5, 'Dropout rate')
# tf.flags.DEFINE_float('neg_weight', 0.1, 'Negative link prediction weight')
# tf.flags.DEFINE_float('weight_decay', 5e-5, 'Weight for L2 loss on embedding matri')
# tf.flags.DEFINE_integer('hidden_dim', 128, 'Hidden embed size')
# tf.flags.DEFINE_integer('hidden_size', 128, 'LSTM hidden size')
# tf.flags.DEFINE_integer('output_dim', 128, 'Output embed size')
# tf.flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')
# tf.flags.DEFINE_integer('neg_sample_size', 1, 'number of negative samples')
# tf.flags.DEFINE_integer('n_component', 1, 'n_component')
# tf.flags.DEFINE_integer('walk_len', 5, 'Walk len')
# tf.flags.DEFINE_integer('test_freq', 20, 'test_freq')
# tf.flags.DEFINE_integer('batch_size', 242, 'batch_size')
# tf.flags.DEFINE_float('max_gradient_norm', 1.0, 'Clip gradients to this norm')
#       
# # GAT parameters
# tf.flags.DEFINE_integer('num_heads', 8, 'Number of heads in each GAT layer')


