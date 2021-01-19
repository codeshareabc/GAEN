from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

FLAGS = tf.flags.FLAGS

class NodeMinibatchIterator(object):
    """
    This minibatch iterator iterates over nodes to sample context pairs for a batch of nodes.

    graphs -- list of networkx graphs
    features -- list of (scipy) sparse node attribute matrices
    adjs -- list of adj matrices (of the graphs)
    phs -- standard tensorflow phs object for feeding
    num_time_steps -- number of graphs to train +1
    context_pairs -- list of (target, context) pairs obtained from random walk sampling.
    batch_size -- size of the minibatches (# nodes)
    """
    def __init__(self, graphs, features, adjs, phs, num_time_steps, p_covss=None, 
                 context_pairs=None, num_features_nonzeros=None, batch_size=100, use_pattern=False):

        self.graphs = graphs
        self.features = features
        self.adjs = adjs
        self.p_covss = p_covss
        self.phs = phs
        self.batch_size = batch_size
        self.batch_num = 0
        self.num_time_steps = num_time_steps
        self.degs = self.construct_degs()
        self.context_pairs = context_pairs
        self.num_features_nonzeros = num_features_nonzeros
        self.max_positive = FLAGS.neg_sample_size
        self.train_nodes = self.graphs[num_time_steps-1].nodes() # all nodes in the graph.
        self.use_pattern = use_pattern
        print ("# train nodes", len(self.train_nodes))

    def construct_degs(self):
        """ Compute node degrees in each graph snapshot."""
        degs = []
        for i in range(0, self.num_time_steps):
            G = self.graphs[i]
            deg = np.zeros((len(G.nodes()),))
            for nodeid in G.nodes():
                neighbors = np.array(list(G.neighbors(nodeid)))
                deg[nodeid] = len(neighbors)
            degs.append(deg)
        return degs

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_nodes)

    def batch_feed_dict(self, batch_nodes):
        """ Feed dict with (a) node pairs, (b) list of attribute matrices (c) list of snapshot adjs and metadata"""
        node_1_all = []
        node_2_all = []
        for t in range(self.num_time_steps):
            node_1 = []
            node_2 = []
            for n in batch_nodes:
                if len(self.context_pairs[t][n]) > self.max_positive:
                    node_1.extend([n]* self.max_positive)
                    node_2.extend(np.random.choice(self.context_pairs[t][n], self.max_positive, replace=False))
                else:
                    if len(self.context_pairs[t][n]) == 0:
                        continue
                    node_1.extend([n]* len(self.context_pairs[t][n]))
                    node_2.extend(self.context_pairs[t][n])

            assert len(node_1) == len(node_2)
            assert len(node_1) <= self.batch_size * self.max_positive
            feed_dict = dict()
            node_1_all.append(node_1)
            node_2_all.append(node_2)

        feed_dict.update({self.phs['node_1'][t]:node_1_all[t] for t in range(self.num_time_steps)})
        feed_dict.update({self.phs['node_2'][t]:node_2_all[t] for t in range(self.num_time_steps)})
        feed_dict.update({self.phs['feats'][t]:self.features[t] for t in range(self.num_time_steps)})
        feed_dict.update({self.phs['adjs'][t]: self.adjs[t] for t in range(self.num_time_steps)})
        feed_dict.update({self.phs['num_features_nonzeros'][t]: self.num_features_nonzeros[t] for t in range(self.num_time_steps)})
        
        if self.use_pattern:
            feed_dict.update({self.phs['p_covss'][t]: self.p_covss[t] for t in range(self.num_time_steps)})
        
        feed_dict.update({self.phs['batch_nodes']:np.array(batch_nodes).astype(np.int32)})
        return feed_dict

    def num_training_batches(self):
        """ Compute the number of training batches (using batch size)"""
        return len(self.train_nodes) // self.batch_size + 1

    def next_minibatch_feed_dict(self):
        """ Return the feed_dict for the next minibatch (in the current epoch) with random shuffling"""
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_nodes))
        batch_nodes = self.train_nodes[start_idx : end_idx]
        return self.batch_feed_dict(batch_nodes)

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_nodes = np.random.permutation(self.train_nodes)
        self.batch_num = 0

    def test_reset(self):
        """ Reset batch number"""
        self.train_nodes =  self.graphs[self.num_time_steps-1].nodes()
        self.batch_num = 0