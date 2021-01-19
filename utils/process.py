from __future__ import print_function
from utils.utilities import run_random_walks_n2v
import numpy as np
import networkx as nx
import scipy.sparse as sp
import pickle
import tensorflow as tf
import os
import csv
import dill
import random
from tensorly.decomposition import non_negative_parafac
import tensorly.contrib.sparse as stl; 
import sparse
from tensorly.contrib.sparse.decomposition import parafac as sparse_parafac
from tensorly.contrib.sparse.kruskal_tensor import kruskal_to_tensor
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


FLAGS = tf.flags.FLAGS
np.random.seed(123)

def dot_mat(x, y, sparse=False):
    if sparse:
        return tf.sparse_tensor_dense_matmul(x, y)
    return tf.matmul(x, y)

def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)

def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def sparse_to_tuple(sparse_mx):
    """Convert scipy sparse matrix to tuple representation (for tf feed dict)."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape
 
    def to_tuple_list(matrices):
        # Input is a list of matrices.
        coords = []
        values = []
        shape = [len(matrices)]
        for i in range(0, len(matrices)):
            mx = matrices[i]
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            # Create proper indices - coords is a numpy array of pairs of indices.
            coords_mx = np.vstack((mx.row, mx.col)).transpose()
            z = np.array([np.ones(coords_mx.shape[0]) * i]).T
            z = np.concatenate((z, coords_mx), axis=1)
            z = z.astype(int)
            coords.extend(z)
            values.extend(mx.data)
 
        shape.extend(matrices[0].shape)
        shape = np.array(shape).astype("int32")
        values = np.array(values).astype("float32")
        coords = np.array(coords)
        return coords, values, shape
 
    if isinstance(sparse_mx, list) and isinstance(sparse_mx[0], list):
        # Given a list of lists, convert it into a list of tuples.
        for i in range(0, len(sparse_mx)):
            sparse_mx[i] = to_tuple_list(sparse_mx[i])
 
    elif isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
 
    return sparse_mx
 
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def transform_school(dataset):
    idfile = os.path.join('datasets/{}'.format(dataset),'ids.txt')
    netfile = os.path.join('datasets/{}'.format(dataset),'primaryschool.csv')
    
    people_id = {}
    people_class = {}
    interacts = {}
    num_node = 0 
    u_labs = []
    with open(idfile, 'r') as idr, open(netfile, 'r') as netr:
        for line in idr:
            params = line.split()
            if params[0] not in people_id:
                people_id[params[0]] = len(people_id)
                if params[1] not in u_labs:
                    u_labs.append(params[1])
                people_class[people_id[params[0]]] = len(u_labs) - 1
                num_node+=1
                
        
        spamreader = csv.reader(netr)
        for row in spamreader:
            items = row[0].split('\t')
            if int(items[0]) not in interacts.keys():
                interacts[int(items[0])] = [(people_id[items[1]],people_id[items[2]])]
            else:
                interacts[int(items[0])].append((people_id[items[1]],people_id[items[2]]))
    
    labels = np.zeros([num_node,len(set(people_class.values()))])
    print(labels.shape)
    for peo in people_class.keys():
        labels[peo,people_class[peo]] = 1 
    
    num_layer = 40
    duration = 13 # all people build links if they have interations within each 13-mins interval
    start_time = list(interacts.keys())[0]
    adjs = []
    for i in range(num_layer):
        base_net = np.zeros([num_node,num_node])   
        end_time = start_time + duration * 60
        for time in interacts.keys():
            if time in range(start_time, end_time):
                for link in interacts[time]:
                    node_i = link[0]
                    node_j = link[1]
                    base_net[node_i,node_j] = 1
                    base_net[node_j,node_i] = 1
        adjs.append(base_net)
        start_time = end_time
    
    graphs = []
    for g in adjs:
        graphs.append(nx.from_numpy_matrix(g))
    
    node_lists = []
    feat_lists = []
    for G in graphs:
        node_lists.append(G.nodes(data=True))
        feat_lists.append(G.edges(data=True))    
#     adj_matrices = []
#     for i in range(adjs.shape[0]):
#         adj_matrices.append(sp.coo_matrix(adjs[i]))
        
    out_file = 'datasets/{}/{}'.format(dataset, 'graphs.pkl')
    with open(out_file, 'wb') as output:
        pickle.dump([node_lists, feat_lists, labels], output)
        
# transform_school('school')   
        
# def transform_data(dataset):
#     graphs = np.load('datasets/{}/{}'.format(dataset, 'graphs.npz'), allow_pickle=True, encoding='bytes')['graph']
#     node_lists = []
#     feat_lists = []
#     for G in graphs:
#         node_lists.append(G.nodes(data=True))
#         feat_lists.append(G.edges(data=True))
#     out_file = 'datasets/{}/{}'.format(dataset, 'graphs.pkl')
#     with open(out_file, 'wb') as output:
#         pickle.dump([node_lists, feat_lists], output)
        
# transform_data("brain")

def load_graphs_classification(dataset):
    file = np.load('datasets/{}/{}'.format(dataset, 'graphs.npz'))
    Features  = file['attmats'] #(n_node, n_time, att_dim)
    Features = np.transpose(Features,[1,0,2])
    labels    = file['labels']  #(n_node, num_classes)
    adjs    = file['adjs']    #(n_time, n_node, n_node)
    print(Features.shape, labels.shape, adjs.shape)
    graphs = []
    for g in adjs:
        graphs.append(nx.from_numpy_matrix(g))
    adj_matrices = []
    features = []
    for i in range(adjs.shape[0]):
        adj_matrices.append(sp.coo_matrix(adjs[i]))
        features.append(sp.coo_matrix(Features[i]))
        
    return graphs, list(adj_matrices), features, labels

# load_graphs_classification("DBLP")

def load_graphs_school(dataset):
    with open('datasets/{}/{}'.format(dataset, 'graphs.pkl'), 'rb') as input:
        node_lists, edge_lists, labels = pickle.load(input,encoding="bytes")
        graphs = []
        for nodes, edges in zip(node_lists, edge_lists):
            G = nx.MultiGraph() 
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)
            graphs.append(G)
    adj_matrices = map(lambda x: nx.adjacency_matrix(x), graphs)
    return graphs, list(adj_matrices), labels

def load_graphs(dataset):
    with open('datasets/{}/{}'.format(dataset, 'graphs.pkl'), 'rb') as input:
        node_lists, edge_lists = pickle.load(input,encoding="bytes")
        graphs = []
        for nodes, edges in zip(node_lists, edge_lists):
            G = nx.MultiGraph() 
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)
            graphs.append(G)
    adj_matrices = map(lambda x: nx.adjacency_matrix(x), graphs)
    return graphs, list(adj_matrices)

# graphs, adj_matrices = load_graphs('school')
# k = 0
# for mat in adj_matrices:
#     mat_ = mat.toarray()
#     k+=np.count_nonzero(mat_)
# print(k)

# G = adj_matrices[0]
# print(G)
#  
# def load_feats(dataset):
#     """ Load node attribute snapshots given the name of dataset"""
#     features = np.load("data/{}/{}".format(dataset, "features.npz"), allow_pickle=True)['feats']
#     print("Loaded {} X matrices ".format(len(features)))
#     return features

def normalize_graph_gcn(adj):
    """GCN-based normalization of adjacency matrix (scipy sparse format). Output is in tuple format"""
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized.todense(), sparse_to_tuple(adj_normalized)

def normalize_pattern(p_covs):
    """GCN-based normalization of adjacency matrix (scipy sparse format). Output is in tuple format"""
    p_covs = sp.coo_matrix(p_covs)
    return p_covs.todense(), sparse_to_tuple(p_covs)


def get_context_pairs(graphs, num_time_steps):
    """ Load/generate context pairs for each snapshot through random walk sampling."""
    load_path = "datasets/{}/train_pairs_n2v_{}.pkl".format(FLAGS.dataset, str(num_time_steps - 2))
    try:
        context_pairs_train = dill.load(open(load_path, 'rb'))
        print("Loaded context pairs from pkl file directly")
    except (IOError, EOFError):
        print("Computing training pairs ...")
        context_pairs_train = []
        for i in range(0, num_time_steps):
            context_pairs_train.append(run_random_walks_n2v(graphs[i], graphs[i].nodes()))
        dill.dump(context_pairs_train, open(load_path, 'wb'))
        print ("Saved pairs")

    return context_pairs_train
    

def get_evaluation_data_classification(adjs, num_time_steps, dataset):   
    """ Load train/val/test examples to evaluate node classification performance"""
    
    
    
    
 
def get_evaluation_data(adjs, num_time_steps, dataset):
    """ Load train/val/test examples to evaluate link prediction performance"""
    eval_idx = num_time_steps - 2
    eval_path = "datasets/{}/eval_{}.npz".format(dataset, str(eval_idx))
    try:
        train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
            np.load(eval_path, encoding='bytes', allow_pickle=True)['data']
        print("Loaded eval data")
    except IOError:
        next_adjs = adjs[eval_idx + 1]
        print("Generating and saving eval data ....")
        train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
            create_data_splits(adjs[eval_idx], next_adjs, val_mask_fraction=0.2, test_mask_fraction=0.6)
        np.savez(eval_path, data=np.array([train_edges, train_edges_false, val_edges, val_edges_false,
                                           test_edges, test_edges_false]))
 
    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false

def get_evaluation_classification_data(dataset, num_nodes, num_time_steps):
    eval_idx = num_time_steps - 2
    eval_path = "datasets/{}/eval_class_{}.npz".format(dataset, str(eval_idx))
    try:
        datas = np.load(eval_path, encoding='bytes', allow_pickle=True)['data']
    except IOError:
        train_ratios = [0.3, 0.5, 0.7]
        datas = []
        for ratio in train_ratios:
            data = []
            for i in range(40):
                idx_val = random.sample(range(num_nodes), int(num_nodes*0.25))
                remaining = np.setdiff1d(np.array(range(num_nodes)), idx_val)
                idx_train = random.sample(list(remaining), int(num_nodes*ratio))
                idx_test = np.setdiff1d(np.array(remaining), idx_train)
                
                data.append([idx_train, idx_val, list(idx_test)])
            datas.append(data)
        np.savez(eval_path, data=np.array(datas))
    return datas

# data = get_evaluation_classification_data('school', 100, 40)
# print(data[0][0])
# print(data[0][1])
# print(data[0][2])
  
def create_data_splits(adj, next_adj, val_mask_fraction=0.2, test_mask_fraction=0.6):
    """In: (adj, next_adj) along with test and val fractions. For link prediction (on all links), all links in
    next_adj are considered positive examples.
    Out: list of positive and negative pairs for link prediction (train/val/test)"""
    edges_all = sparse_to_tuple(next_adj)[0]  # All edges in original adj.
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)  # Remove diagonal elements
    adj.eliminate_zeros()
    assert np.diag(adj.todense()).sum() == 0
    if next_adj is None:
        raise ValueError('Next adjacency matrix is None')
 
    edges_next = np.array(list(set(nx.from_scipy_sparse_matrix(next_adj).edges())))
    edges = []   # Constraint to restrict new links to existing nodes.
    for e in edges_next:
        if e[0] < adj.shape[0] and e[1] < adj.shape[0]:
            edges.append(e)
    edges = np.array(edges)
 
    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)
 
    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    num_test = int(np.floor(edges.shape[0] * test_mask_fraction))
    num_val = int(np.floor(edges.shape[0] * val_mask_fraction))
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
 
    # Create train edges.
    train_edges_false = []
    while len(train_edges_false) < len(train_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue
        if train_edges_false:
            if ismember([idx_j, idx_i], np.array(train_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(train_edges_false)):
                continue
        train_edges_false.append([idx_i, idx_j])
 
    # Create test edges.
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])
 
    # Create val edges.
    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue
 
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])
 
    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)
    print("# train examples: ", len(train_edges), len(train_edges_false))
    print("# val examples:", len(val_edges), len(val_edges_false))
    print("# test examples:", len(test_edges), len(test_edges_false))
 
    return list(train_edges), train_edges_false, list(val_edges), val_edges_false, list(test_edges), test_edges_false
     
def testcp():
    shape = (1000, 1000, 1000)
    rank = 1000
    starting_weights = stl.ones((rank))
    starting_factors = [sparse.random((i, rank)) for i in shape]
    tensor = kruskal_to_tensor((starting_weights, starting_factors))
    import time
    t = time.time()
    sparse_kruskal = sparse_parafac(tensor, 5, init='random')
    print(time.time() - t)
# testcp()
       
def cal_patterns(adjs, num_time_steps, n_component):  
    
    load_path = "datasets/{}/node_pattern_{}.pkl".format(FLAGS.dataset, str(num_time_steps - 2))
    try:
        p_covss = dill.load(open(load_path, 'rb'))
        print("Loaded dynamic node patterns from pkl file directly")
    except (IOError, EOFError):
        print("Computing dynamic node patterns ...")
        p_covss=[]
        for i in range(num_time_steps):
            tensor = []
            for j in range(i+1):
                tensor.append(adjs[j].todense().getA())
#             A, C = non_neg_parafac_sparse(tensor, n_component)
            A, C = non_neg_parafac(tensor, n_component)
            
            A_p = np.matmul(A, C.T)
            p_covs = np.matmul(A_p, A_p.T)
            
            p_covss.append(p_covs)
        dill.dump(p_covss, open(load_path, 'wb'))    
    
    return p_covss
       
def non_neg_parafac_sparse(tensor, n_component):
    starting_factors = []
#     for ten in tensor:
#         ten = ten.astype(float)
#         stensor_ = sp.csr_matrix(ten)
# #         stensor_= sparse.COO(ten)
#         starting_factors.append(sparse.random(stensor_.shape))
#         print(stensor_.shape)
    starting_factors= [sparse.random(stensor_.shape) for stensor_ in tensor]     
    starting_weights = stl.ones(np.array(tensor).shape[2])
#     print(starting_factors)
    tensor_ = kruskal_to_tensor((starting_weights, starting_factors))
#     print(tensor_.fill_value)
#     print(len(tensor))
    _, factors = sparse_parafac(tensor_, n_component, init='random')
    
    C = np.array(factors[0].todense())
    A = np.array(factors[1].todense())
#     B = np.array(factors[2]) 
    
    return A, C    

def non_neg_parafac(tensor, n_component):
    tensor = np.array(tensor).astype(float)
    print(tensor.shape)
    _, factors = non_negative_parafac(tensor, rank=n_component, init='random')
    
    C = np.array(factors[0])
    A = np.array(factors[1])
#     B = np.array(factors[2]) 
    return A, C    
