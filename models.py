from utils.process import *
from tensorflow.python.ops import math_ops


class Sp_GCN(object):
    def __init__(self, in_dim, out_dim, act, dropout_prob, num_features_nonzero,
                 dropout=False, bias=True, name='gcn'):
        self.act = act
        self.bias = bias
        self.dropout = dropout
        self.var = {}
        with tf.variable_scope(name):
            self.var['w']=glorot([in_dim, out_dim], name='w')
            if self.bias:
                self.var['b']=zeros([out_dim], name='b')

            if self.dropout:
                self.dropout_prob = dropout_prob
            else:
                self.dropout_prob = 0.
            self.num_features_nonzero = num_features_nonzero           

    def call(self, adj_norm, x, sparse=False):
        if sparse:
            x = sparse_dropout(x, 1-self.dropout_prob, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout_prob)
        hw = dot_mat(x=x, y=self.var['w'], sparse=sparse)  
        ahw = dot_mat(x=adj_norm, y=hw, sparse=True)
        embed_out = self.act(ahw)
        if self.bias:
            embed_out = self.act(tf.add(ahw, self.var['b']))   
            
        return embed_out  
    
    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)    
        
        
class GAT(object):
    
    def __init__(self, input_dim, output_dim, n_heads, attn_drop, ffd_drop, act=tf.nn.elu, residual=False,
                 bias=True, sparse_inputs=False, name=''):
        self.attn_drop = attn_drop
        self.ffd_drop = ffd_drop
        self.act = act
        self.bias = bias
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.residual = residual
        self.sparse_inputs = sparse_inputs  
        self.n_calls = 0  
        self.name = name     
        
    def call(self, adj_norm, x):
        self.n_calls += 1
        adj = adj_norm
        attentions = []
        reuse_scope = None
        for j in range(self.n_heads):
            if self.n_calls > 1:
                reuse_scope = True

            attentions.append(self.sp_attn_head(seq=x, adj_mat=adj, in_sz=self.input_dim,
                                                out_sz=self.output_dim, activation=self.act,
                                                in_drop=self.ffd_drop, coef_drop=self.attn_drop, residual=self.residual,
                                                layer_str="l_{}_h_{}".format(self.name, j),
                                                sparse_inputs=self.sparse_inputs,
                                                reuse_scope=reuse_scope))        
        
#         h = tf.concat(attentions, axis=-1)
#         return h
        logits = tf.add_n(attentions) / self.n_heads
        return logits

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs) 

    @staticmethod
    def leaky_relu(features, alpha=0.2):
        return math_ops.maximum(alpha * features, features)

    def sp_attn_head(self, seq, in_sz, out_sz, adj_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False,
                     layer_str="", sparse_inputs=False, reuse_scope=None):
        """ Sparse Attention Head for the GAT layer. Note: the variable scope is necessary to avoid
        variable duplication across snapshots"""

        with tf.variable_scope('sp_attn', reuse=reuse_scope):
            if sparse_inputs:
                weight_var = tf.get_variable("layer_" + str(layer_str) + "_weight_transform", shape=[in_sz, out_sz],
                                             dtype=tf.float32)
                seq_fts = tf.expand_dims(tf.sparse_tensor_dense_matmul(seq, weight_var), axis=0)  # [N, F]
                
            else:
                seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False,
                                           name='layer_' + str(layer_str) + '_weight_transform', reuse=reuse_scope)

            # Additive self-attention.
            f_1 = tf.layers.conv1d(seq_fts, 1, 1, name='layer_' + str(layer_str) + '_a1', reuse=reuse_scope)
            f_2 = tf.layers.conv1d(seq_fts, 1, 1, name='layer_' + str(layer_str) + '_a2', reuse=reuse_scope)
            f_1 = tf.reshape(f_1, [-1, 1])  # [N, 1]
            f_2 = tf.reshape(f_2, [-1, 1])  # [N, 1]

            logits = tf.sparse_add(adj_mat * f_1, adj_mat * tf.transpose(f_2))  # adj_mat is [N, N] (sparse)

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

#             if residual:
#                 residual_wt = tf.get_variable("layer_" + str(layer_str) + "_residual_weight", shape=[in_sz, out_sz],
#                                               dtype=tf.float32)
#                 if sparse_inputs:
#                     ret = ret + tf.expand_dims(tf.sparse_tensor_dense_matmul(seq, residual_wt),
#                                                axis=0)  # [N, F] * [F, D] = [N, D].
#                 else:
#                     ret = ret + tf.layers.conv1d(seq, out_sz, 1, use_bias=False,
#                                                  name='layer_' + str(layer_str) + '_residual_weight', reuse=reuse_scope)
            return activation(ret)             
        
        
class Sp_GCN_GRU(object):
    def __init__(self, num_layers, hidden_size, seq_len, input_dim, 
                 hidden_dim, output_dim, act, dropout_prob, num_features_nonzeros):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.act = act
        self.dropout_prob = dropout_prob
        self.num_features_nonzeros = num_features_nonzeros

        self.w_list = []
        for i in range(self.num_layers):
            if i==0:
                w_i = glorot([input_dim, hidden_dim], name='w'+str(i))
            else:
                w_i = glorot([hidden_dim, output_dim], name='w'+str(i))
            self.w_list.append(w_i)
        
        
    def call(self, adjs, feats, sparse=False): 
        
        last_l_seq=[]
        for i in range(self.seq_len):
            adj_norm = adjs[i]
            x = feats[i]
            
            if sparse:
                x = sparse_dropout(x, 1-self.dropout_prob, self.num_features_nonzeros[i])
            else:
                x = tf.nn.dropout(x, 1-self.dropout_prob)
            hw = dot_mat(x=x, y=self.w_list[0], sparse=sparse)  
            ahw = dot_mat(x=adj_norm, y=hw, sparse=True)
            last_l = self.act(ahw)
            
#             last_l = self.act(adj.matmul(feat.matmul(self.w_list[0])))
            for j in range(1, self.num_layers):
                hw = dot_mat(x=last_l, y=self.w_list[j])  
                ahw = dot_mat(x=adj_norm, y=hw, sparse=True)
                last_l = self.act(ahw)
#                 last_l = self.act(adj.matmul(last_l.matmul(self.w_list[i])))
            last_l_seq.append(last_l)            
            
        last_l_seq = tf.stack(last_l_seq)    
#         rnn_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, 
#                                            initializer=tf.keras.initializers.glorot_normal)
        rnn_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size, 
                                           kernel_initializer=tf.keras.initializers.glorot_normal,
                                           bias_initializer=tf.keras.initializers.glorot_normal)
        rnn_outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell,
                                     inputs=last_l_seq,
                                     dtype=tf.float32)
        return rnn_outputs
            
    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)             
            
class Sp_GCN_LSTM_B(object):
    def __init__(self, num_layers, hidden_size, seq_len, input_dim, 
                 hidden_dim, output_dim, act, dropout_prob, num_features_nonzeros):
        assert num_layers==2, 'a two-layer GCN is required'
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.act = act
        self.dropout_prob = dropout_prob
        self.num_features_nonzeros = num_features_nonzeros

        self.w_list = []
        for i in range(self.num_layers):
            if i==0:
                w_i = glorot([input_dim, hidden_dim], name='w'+str(i))
            else:
                w_i = glorot([hidden_dim, output_dim], name='w'+str(i))
            self.w_list.append(w_i)        
        
    def call(self, adjs, feats, sparse=False):   
        last_l_seq = []
        last_2_seq = []
        for i in range(self.seq_len):
            adj_norm = adjs[i]
            x = feats[i]
            if sparse:
                x = sparse_dropout(x, 1-self.dropout_prob, self.num_features_nonzeros[i])
            else:
                x = tf.nn.dropout(x, 1-self.dropout_prob)
            hw = dot_mat(x=x, y=self.w_list[0], sparse=sparse)  
            ahw = dot_mat(x=adj_norm, y=hw, sparse=True)
            l1_seq = self.act(ahw)
            last_l_seq.append(l1_seq)
        with tf.variable_scope('lstm1'):    
            last_l_seq = tf.stack(last_l_seq)
            l1_rnn_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, 
                                               initializer=tf.keras.initializers.glorot_normal)
            l1_rnn_outputs, _ = tf.nn.dynamic_rnn(cell=l1_rnn_cell,
                                         inputs=last_l_seq,
                                         dtype=tf.float32)        
            
        sparse = False
        for i in range(self.seq_len):
            adj_norm = adjs[i]
            x = l1_rnn_outputs[i]
            hw = dot_mat(x=x, y=self.w_list[1], sparse=sparse)  
            ahw = dot_mat(x=adj_norm, y=hw, sparse=True)
            l2_seq = self.act(ahw)
            last_2_seq.append(l2_seq)  
            
        with tf.variable_scope('lstm2'):        
            last_2_seq = tf.stack(last_2_seq)
            l2_rnn_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, 
                                               initializer=tf.keras.initializers.glorot_normal)
            l2_rnn_outputs, _ = tf.nn.dynamic_rnn(cell=l2_rnn_cell,
                                         inputs=last_2_seq,
                                         dtype=tf.float32)  
        
        return l2_rnn_outputs 

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)      
        
        
class Sp_GAT_GRU(object):
    def __init__(self, hidden_size, seq_len, input_dim, output_dim, n_heads, attn_drop, 
                 ffd_drop, act=tf.nn.elu, residual=False,
                 bias=True, sparse_inputs=False, name=''):
        self.model = GAT(input_dim, output_dim, n_heads, attn_drop, ffd_drop, act, 
                         residual, bias, sparse_inputs, name)
        
        self.hidden_size = hidden_size
        self.seq_len = seq_len   
             
        
    def call(self, adjs, feats):     
        last_l_seq=[]    
        for i in range(self.seq_len):
            adj = adjs[i]
            feat = feats[i]    
            output = self.model(adj, feat)        
            embeds = tf.reshape(output, [-1, FLAGS.output_dim])
            last_l_seq.append(embeds)    
        
        last_l_seq = tf.stack(last_l_seq)    

#             rnn_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, 
#                                                initializer=tf.keras.initializers.glorot_normal)
        rnn_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size, 
                                       kernel_initializer=tf.keras.initializers.glorot_normal,
                                       bias_initializer=tf.keras.initializers.glorot_normal)
        rnn_outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell,
                                     inputs=last_l_seq,
#                                      sequence_length=self.seq_len,
                                     dtype=tf.float32)
        return rnn_outputs
            
    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs) 
        
