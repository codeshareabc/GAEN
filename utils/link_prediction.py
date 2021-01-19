from __future__ import division, print_function
from sklearn.metrics import roc_auc_score, roc_curve, auc
import numpy as np
from sklearn import linear_model
from collections import defaultdict
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn import svm
import statistics

np.random.seed(123)
operatorTypes = ["HAD"]


def write_to_csv(test_results, output_name, model_name, dataset, time_steps, mod='val'):
    """Output result scores to a csv file for result logging"""
    with open(output_name, 'a+') as f:
        for op in test_results:
            print("{} results ({})".format(model_name, mod), test_results[op])
            _, best_auc = test_results[op]
            f.write("{},{},{},{},{},{},{}\n".format(dataset, time_steps, model_name, op, mod, "AUC", best_auc))


def get_link_score(fu, fv, operator):
    """Given a pair of embeddings, compute link feature based on operator (such as Hadammad product, etc.)"""
    fu = np.array(fu)
    fv = np.array(fv)
    if operator == "HAD":
        return np.multiply(fu, fv)
    else:
        raise NotImplementedError


def get_link_feats(links, source_embeddings, target_embeddings, operator):
    """Compute link features for a list of pairs"""
    features = []
    for l in links:
        a, b = l[0], l[1]
        f = get_link_score(source_embeddings[a], target_embeddings[b], operator)
        features.append(f)
    return features


def get_random_split(train_pos, train_neg, val_pos, val_neg, test_pos, test_neg):
    """ Randomly split a given set of train, val and test examples"""
    all_data_pos = []
    all_data_neg = []

    all_data_pos.extend(train_pos)
    all_data_neg.extend(train_neg)
    all_data_pos.extend(test_pos)
    all_data_neg.extend(test_neg)

    # re-define train_pos, train_neg, test_pos, test_neg.
    random.shuffle(all_data_pos)
    random.shuffle(all_data_neg)

    train_pos = all_data_pos[:int(0.2 * len(all_data_pos))]
    train_neg = all_data_neg[:int(0.2 * len(all_data_neg))]

    test_pos = all_data_pos[int(0.2 * len(all_data_pos)):]
    test_neg = all_data_neg[int(0.2 * len(all_data_neg)):]
    print("# train :", len(train_pos) + len(train_neg), "# val :", len(val_pos) + len(val_neg),
          "#test :", len(test_pos) + len(test_neg))
    return train_pos, train_neg, val_pos, val_neg, test_pos, test_neg

def evaluate_node_classification(emb, labels, datas):
    
#     train_ratios = [0.3, 0.5, 0.7]
#     num_iterate = 10
#     num_nodes = labels.shape[0]
#     
    labels = np.argmax(labels,1)
    
    average_accs = []
#     stdev_accs = []
    
#     idx_val = random.sample(range(num_nodes), int(num_nodes*0.25))
#     remaining = np.setdiff1d(np.array(range(num_nodes)), idx_val)
    for train_nodes in datas:
        temp_accs = []
        for train_node in train_nodes:
            
            train_vec = emb[train_node[0]]
            train_y = labels[train_node[0]]
            val_vec = emb[train_node[1]]
            val_y = labels[train_node[1]]
            test_vec = emb[train_node[2]]
            test_y = labels[train_node[2]]
            
            clf = LogisticRegression(multi_class='auto',solver='lbfgs',  max_iter=4000)
#             clf = LogisticRegression(multi_class='auto',solver='lbfgs')
            clf.fit(train_vec, train_y)
            
            y_pred = clf.predict(test_vec)
            
            acc = accuracy_score(test_y, y_pred) 
            temp_accs.append(acc)
        
        average_acc = statistics.mean(temp_accs)
        average_accs.append(average_acc)
#         stdev_acc = statistics.stdev(temp_accs)
    
    return average_accs
   
    
    
    # clf = svm.SVC(decision_function_shape='ovo', kernel='linear', probability=True)
          

def evaluate_classifier(train_pos, train_neg, val_pos, val_neg, test_pos, test_neg, source_embeds, target_embeds):
    """Downstream logistic regression classifier to evaluate link prediction"""
    test_results = []
    val_results = []

    test_pred_true = []
    val_pred_true = []

    for operator in operatorTypes:
        train_pos_feats = np.array(get_link_feats(train_pos, source_embeds, target_embeds, operator))
        train_neg_feats = np.array(get_link_feats(train_neg, source_embeds, target_embeds, operator))
        val_pos_feats = np.array(get_link_feats(val_pos, source_embeds, target_embeds, operator))
        val_neg_feats = np.array(get_link_feats(val_neg, source_embeds, target_embeds, operator))
        test_pos_feats = np.array(get_link_feats(test_pos, source_embeds, target_embeds, operator))
        test_neg_feats = np.array(get_link_feats(test_neg, source_embeds, target_embeds, operator))

        train_pos_labels = np.array([1] * len(train_pos_feats))
        train_neg_labels = np.array([-1] * len(train_neg_feats))
        val_pos_labels = np.array([1] * len(val_pos_feats))
        val_neg_labels = np.array([-1] * len(val_neg_feats))

        test_pos_labels = np.array([1] * len(test_pos_feats))
        test_neg_labels = np.array([-1] * len(test_neg_feats))
        train_data = np.vstack((train_pos_feats, train_neg_feats))
        train_labels = np.append(train_pos_labels, train_neg_labels)

        val_data = np.vstack((val_pos_feats, val_neg_feats))
        val_labels = np.append(val_pos_labels, val_neg_labels)

        test_data = np.vstack((test_pos_feats, test_neg_feats))
        test_labels = np.append(test_pos_labels, test_neg_labels)

        logistic = linear_model.LogisticRegression(solver='lbfgs')
        logistic.fit(train_data, train_labels)
        test_predict = logistic.predict_proba(test_data)[:, 1]
        val_predict = logistic.predict_proba(val_data)[:, 1]
            
        val_predict_bin = np.where(val_predict > 0.5, 1, -1)
        val_auc_micro = roc_auc_score(val_labels, val_predict, average='micro')
        val_auc_macro = roc_auc_score(val_labels, val_predict, average='macro')
        val_precision = precision_score(val_labels, val_predict_bin, average='macro')

        test_predict_bin = np.where(test_predict > 0.5, 1, -1)
        test_auc_micro = roc_auc_score(test_labels, test_predict, average='micro')
        test_auc_macro = roc_auc_score(test_labels, test_predict, average='macro')
        test_precision = precision_score(test_labels, test_predict_bin, average='macro')
        
        val_results.extend([val_auc_micro, val_auc_macro, val_precision])
        test_results.extend([test_auc_micro, test_auc_macro, test_precision])

#         val_pred_true.extend(zip(val_predict, val_labels))
#         test_pred_true.extend(zip(test_predict, test_labels))

    return val_results, test_results