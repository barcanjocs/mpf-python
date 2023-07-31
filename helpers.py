import torch
from sklearn.metrics import precision_recall_curve, auc
import numpy as np 
import csv

sf = torch.nn.Softmax(dim=-1)

def short_pr_auc(new_preds, preds_scores, margin) :

    selected_probs = sf(torch.from_numpy(np.array(preds_scores))).float()

    binary_truth = np.zeros(len(new_preds))
    for i in range(0, len(new_preds)) :
        
            if abs(i - new_preds[i]) <= margin :
                binary_truth[i] = 1

    precision, recall, _ = precision_recall_curve(binary_truth, selected_probs)
    pr_auc = auc(recall, precision)
    return precision, recall, pr_auc


def custom_label_pr_auc(new_preds, preds_scores, labels,  margin) :
    selected_probs = preds_scores
    
    binary_truth = np.zeros(len(new_preds))
    for i in range(0, len(new_preds)) :
            if abs(labels[i] - new_preds[i]) <= margin :
                binary_truth[i] = 1

    precision, recall, _ = precision_recall_curve(binary_truth, selected_probs)
    pr_auc = auc(recall, precision)
    return precision, recall, pr_auc

def pr_auc_to_file(precision, recall, auc, filename, mode = "w", title = None) :
    with open(filename, mode) as out_file:
        writer = csv.writer(out_file)
        
        if title:
            writer.writerow(title)

        writer.writerow(list(precision))
        writer.writerow(list(recall))
        writer.writerow(auc)
        writer.writerow("")

