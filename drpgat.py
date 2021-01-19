from utils.process import *
from tensorflow.python.ops import math_ops
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

class DRPGAT(object):
    """
    act: activation function for GAT
    n_node: number of nodes on the network
    output_dim: output embed size for GAT
    seq_len: number of graphs
    n_heads: number of heads for GAT
    attn_drop: attention/coefficient matrix dropout rate
    ffd_drop: feature matrix dropout rate
    residual: if using short cut or not for GRU network
    """
    def __init__(self, act, n_node, input_dim, output_dim, seq_len,
                 n_heads, attn_drop, ffd_drop, residual=False,
                 bias=True, sparse_inputs=False, name=''):
        self.act = act
        self.n_node = n_node
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.attn_drop = attn_drop
        self.ffd_drop = ffd_drop
        self.residual=residual
        self.bias=bias
        self.sparse_inputs=sparse_inputs
        self.name = name
        
        self.var = {}
        
        self.evolve_weights = GRU(n_node, input_dim, output_dim, n_heads, residual)
        
    def call(self, adjs, feats, p_covss):
        embeds = []
        adj = adjs[0]
        feat = feats[0]
        weight_vars = {}
        
        model = GAT(self.input_dim, self.output_dim, self.n_heads, self.attn_drop, 
                        self.ffd_drop, self.act, self.bias, 
                        self.sparse_inputs, self.name)
        
        for i in range(self.n_heads):
            weight_var = tf.compat.v1.get_variable("layer_" + str(i) + "_weight_transform", shape=[self.input_dim, self.output_dim],
                                            dtype=tf.float32)
            weight_vars[i] = weight_var
            
            self.var['weight_var_'+str(i)] = weight_var
            
        output = model(feat, adj, p_covss[0], weight_vars)
#         print(output.shape)
        embed = tf.reshape(output, [-1, self.output_dim])
        embeds.append(embed)
        
        for i in range(1, self.seq_len):
            adj = adjs[i]
            feat = feats[i]
            
            weight_vars = self.evolve_weights(adj, weight_vars)
            
            output = model(feat, adj, p_covss[i], weight_vars)
            embed = tf.reshape(output, [-1, self.output_dim])
            embeds.append(embed)
            
        return embeds
    
    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs) 

class GRU(object):
    def __init__(self, n_node, input_dim, output_dim, n_head, residual=False):
        self.n_node = n_node
        self.n_head = n_head
        self.residual = residual
        self.gru_cell = GRU_cell(self.n_node, input_dim, output_dim)
        
    def call(self, adj_mat, weight_vars):
        weight_vars_next = {}
        for i in range(self.n_head):
            if self.residual:
                new_Q = self.gru_cell(adj_mat, weight_vars[i]) + weight_vars[i]
            else:
                new_Q = self.gru_cell(adj_mat, weight_vars[i])
            weight_vars_next[i] = new_Q     
        return weight_vars_next

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs) 

class GRU_cell(object):
    def __init__(self, n_node, input_dim, output_dim):
        self.n_node = n_node
        
        self.reset  = GRU_gate(n_node, input_dim, output_dim, tf.nn.sigmoid, name='1')
        self.update = GRU_gate(n_node, input_dim, output_dim, tf.nn.sigmoid, name='2')
        self.htilda = GRU_gate(n_node, input_dim, output_dim, tf.nn.tanh, name='3')
        
    def call(self, adj_mat, prev_w):
        reset = self.reset(adj_mat, prev_w)
        update = self.update(adj_mat, prev_w)
        
        h_cap = reset * prev_w
        h_cap = self.htilda(adj_mat, h_cap)
        new_Q = (1 - update) * prev_w + update * h_cap

        return new_Q   

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)        

class GRU_gate(object):
    def __init__(self, n_node, input_dim, output_dim, act, name, reduce=False):
        self.activation = act
        self.name = name
        self.reduce = reduce
        
        with tf.compat.v1.variable_scope(self.name+str('params')):
            self.W = glorot([n_node, output_dim])
            self.U = glorot([output_dim, output_dim])
            self.bias = zeros([input_dim, output_dim])
            if n_node != input_dim:
                self.reduce = True
                self.P = glorot([input_dim, n_node])
        
        
    def call(self, adj_mat, prev_w):
#         out = self.activation(self.W.matmul(x) + \
#                               self.U.matmul(hidden) + \
#                               self.bias)
        with tf.variable_scope(self.name):
            if self.reduce:
                temp_ = dot_mat(x=adj_mat, y=self.W, sparse=True)
                out = self.activation(dot_mat(x=self.P, y=temp_) + \
                                      dot_mat(x=prev_w, y=self.U) + \
                                      self.bias)
            else:
                out = self.activation(dot_mat(x=adj_mat, y=self.W, sparse=True) + \
                                      dot_mat(x=prev_w, y=self.U) + \
                                      self.bias)
        return out

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)   


class GAT(object):
    def __init__(self, input_dim, output_dim, n_heads, attn_drop, ffd_drop, act=tf.nn.elu,
                 bias=True, sparse_inputs=False, name=''):
        self.attn_drop = attn_drop
        self.ffd_drop = ffd_drop
        self.act = act
        self.bias = bias
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sparse_inputs = sparse_inputs  
        self.name = name  
        self.n_calls = 0   
        
        
    def call(self, x, adj_norm, p_covs, att_weights):
        attentions = []
        self.n_calls += 1
        reuse_scope = None
        for j in range(self.n_heads):
            if self.n_calls > 1:
                reuse_scope = True
            attentions.append(self.sp_attn_head(feat=x, p_covs=p_covs, in_sz=self.input_dim, out_sz=self.output_dim,
                                                adj_mat=adj_norm, weight_var=att_weights[j], 
                                                activation=self.act,
                                                in_drop=self.ffd_drop, coef_drop=self.attn_drop,
                                                layer_str=str(j), reuse_scope=reuse_scope, sparse_inputs=True))    
#         h = tf.concat(attentions, axis=-1)
#         return h       
        logits = tf.add_n(attentions) / self.n_heads
        return logits 
        
        
    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs) 

    @staticmethod
    def leaky_relu(features, alpha=0.2):
        return math_ops.maximum(alpha * features, features)

    def sp_attn_head(self, feat, p_covs, in_sz, out_sz, adj_mat, weight_var, activation, 
                     in_drop=0.0, coef_drop=0.0, layer_str='', reuse_scope=None, sparse_inputs=False):
        """ Sparse Attention Head for the GAT layer. Note: the variable scope is necessary to avoid
        variable duplication across snapshots"""
        with tf.variable_scope('sp_drgat', reuse=reuse_scope):
            if sparse_inputs:
                seq_fts = tf.expand_dims(tf.sparse.sparse_dense_matmul(feat, weight_var), axis=0)  # [N, F]
            else:
                seq_fts = tf.layers.conv1d(feat, out_sz, 1, use_bias=False)
            
            
            f_1 = tf.layers.conv1d(seq_fts, 1, 1, name='layer_' + str(layer_str) + '_a1', reuse=reuse_scope)
            f_2 = tf.layers.conv1d(seq_fts, 1, 1, name='layer_' + str(layer_str) + '_a2', reuse=reuse_scope)
            f_1 = tf.reshape(f_1, [-1, 1])  # [N, 1]
            f_2 = tf.reshape(f_2, [-1, 1])  # [N, 1]
    
            logits = tf.sparse_add(adj_mat * f_1, adj_mat * tf.transpose(f_2))  # adj_mat is [N, N] (sparse)
            
            logits = logits * tf.nn.softmax(p_covs)
            
            leaky_relu = tf.SparseTensor(indices=logits.indices,
                                         values=self.leaky_relu(logits.values),
                                         dense_shape=logits.dense_shape)
            coefficients = tf.sparse_softmax(leaky_relu)  # [N, N] (sparse)  
              
            if coef_drop != 0.0:
                coefficients = tf.SparseTensor(indices=coefficients.indices,
                                               values=tf.nn.dropout(coefficients.values, 1.0 - coef_drop),
                                               dense_shape=coefficients.dense_shape)  # [N, N] (sparse)
            if in_drop != 0.0:
                seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)  # [N, D]            
                    
            seq_fts = tf.squeeze(seq_fts)
            values = tf.sparse_tensor_dense_matmul(coefficients, seq_fts)
            values = tf.reshape(values, [-1, out_sz])
            values = tf.expand_dims(values, axis=0)
            ret = values  # [1, N, F]

        return activation(ret)          
        
                    