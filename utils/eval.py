import numpy as np
import tensorflow as tf

def get_link_score(fu, fv, operator):
    """Given a pair of embeddings, compute link feature based on operator (such as Hadammad product, etc.)"""
    fu = np.array(fu)
    fv = np.array(fv)
    if operator == "HAD":
        return tf.multiply(fu, fv)
    else:
        raise NotImplementedError


def get_link_feats(links, node_embeddings, operator):
    """Compute link features for a list of pairs"""
    features = []
    for l in links:
        a, b = l[0], l[1]
        f = get_link_score(node_embeddings[a], node_embeddings[b], operator)
        features.append(f)
    return features

def cal_score_cosine(embeds_pa, embeds_pb, embeds_nb):
    MD_len = tf.sqrt(tf.reduce_sum(tf.multiply(embeds_pa, embeds_pa)))
    PSD_len = tf.sqrt(tf.reduce_sum(tf.multiply(embeds_pb, embeds_pb)))
    NSD_len = tf.sqrt(tf.reduce_sum(tf.matmul(embeds_pa, tf.transpose(embeds_nb))))
    MPSD_dist = tf.sqrt(tf.reduce_sum(tf.multiply(embeds_pa, embeds_pb)))
    MNSD_dist = tf.sqrt(tf.reduce_sum(tf.matmul(embeds_pa, tf.transpose(embeds_nb))))
    
    MPSD_score = tf.div(MPSD_dist, tf.multiply(MD_len, PSD_len))
    MNSD_score = tf.div(MNSD_dist, tf.multiply(MD_len, NSD_len))  
    
    return MPSD_score, MNSD_score

def cal_score(embeds_pa, embeds_pb, embeds_nb):
    pos_score  = tf.reduce_sum(embeds_pa * embeds_pb, axis=1)
    neg_score  = 1-tf.matmul(embeds_pa, tf.transpose(embeds_nb))
    
    return pos_score, neg_score 