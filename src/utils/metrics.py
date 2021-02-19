from typing import Union
import numpy as np
from sklearn import metrics
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.metrics import roc_curve, precision_recall_curve
from scipy.stats import pearsonr, spearmanr
from collections import OrderedDict
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
import logging


def flatten(y_true, y_pred):
    pred_flat = np.argmax(y_pred, axis=1).flatten()
    labels_flat = y_true.flatten()
    return labels_flat, pred_flat

def accuracy(y_true, y_pred):
    labels, preds = flatten(y_true, y_pred)
    return metrics.accuracy_score(labels, preds)

def precision_score(y_true, y_pred):
    labels, preds = flatten(y_true, y_pred)
    return metrics.precision_score(labels, preds)

def recall_score(y_true, y_pred):
    labels, preds = flatten(y_true, y_pred)
    return metrics.recall_score(labels, preds)

def f1_score(y_true, y_pred):
    labels, preds = flatten(y_true, y_pred)
    return metrics.f1_score(labels, preds)

def mean_squared_error(y_true, y_pred):
    labels, preds = flatten(y_true, y_pred)
    return metrics.mean_squared_error(y_true, y_pred)

def root_mse(y_true, y_pred):
    labels, preds = flatten(y_true, y_pred)
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mean_squared_log_error(y_true, y_pred):
    labels, preds = flatten(y_true, y_pred)
    return metrics.mean_squared_log_error(y_true, y_pred)

def r2(y_true, y_pred):
    labels, preds = flatten(y_true, y_pred)
    return metrics.r2_score(y_true, y_pred)

def get_accuracy_and_best_threshold_from_pr_curve(predictions, labels):
    assert 1 in labels and 0 in labels, "Some labels are not present"
    num_pos_class = sum(l for l in labels if l==1)
    num_neg_class = sum(l for l in labels if l==0)
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    tp = recall * num_pos_class
    fp = (tp / precision) - tp
    tn = num_neg_class - fp
    acc = (tp + tn) / (num_pos_class + num_neg_class)

    best_threshold = thresholds[np.argmax(acc)]
    return np.amax(acc), best_threshold

def plot_roc_curve(labels, predictions, savepath=None):
    auroc = metrics.roc_auc_score(labels, predictions)
    fpr, tpr, _ = metrics.roc_curve(labels, predictions)
    plt.plot(fpr, tpr, linestyle='--', label='Best model (Area = %0.2f)' % auroc)
    # Title
    plt.title('ROC Plot')
    # Axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # Show legend
    plt.legend() # 
    # Show plot
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath)

def cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    This function can be used as a faster replacement for 1-scipy.spatial.distance.cdist(a,b)
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = a / a.norm(dim=-1)[:, None]
    b_norm = b / b.norm(dim=-1)[:, None]
    return torch.mm(a_norm, b_norm.transpose(0, 1))


class Metrics:
    """ class that acts as a manager for the metrics used in training and evaluation """
    def __init__(self, *args, mode="training", return_predictions=False):
        """ items are instances of AverageMeter"""
        assert mode in ["training", "validation"]
        self.mode = mode
        self.metrics = []
        for m in args:
            self.metrics.append(m(return_predictions=return_predictions))

    def display_metrics(self):
        for meter in self.metrics:
            print("Current "+self.mode.upper()+" "+str(meter))

    def set_postfix(self):
        postfixes = {}
        for meter in self.metrics:
            postfixes[meter.get_name] = "{:.2f}".format(meter.avg)
        return postfixes 


class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self, name, return_predictions=False):
        self.name = name
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.return_predictions = return_predictions
        if self.return_predictions:
            self.all_predictions = []
            self.all_labels = []

    def __str__(self):
        return f"average {self.name}: {self.avg}" 

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        if self.return_predictions:
            self.all_predictions = []
            self.all_labels = []

    def update(self, val, n=1, save_predictions=False):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def get_name(self):
        return self.name
        

class AccuracyMeter(AverageMeter):

    def __init__(self, **kwargs):
        super().__init__(name="accuracy", **kwargs)

    def __str__(self):
        return f"acc: {self.avg}"

    def update(self, preds, labels, n, **kwargs):
        self.val = accuracy(labels, preds)
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.return_predictions:
            labs, preds = flatten(labels, preds)
            self.all_labels.extend(labs.tolist())
            self.all_predictions.extend(preds.tolist())


class F1Meter(AverageMeter):
    def __init__(self, **kwargs):
        super().__init__(name="f1", **kwargs)

    def __str__(self):
        return f"f1: {self.avg}"

    def update(self, preds, labels, n, **kwargs):
        self.val = f1_score(labels, preds)
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count


class PrecisionMeter(AverageMeter):
    def __init__(self, **kwargs):
        super().__init__(name="pr", **kwargs)

    def __str__(self):
        return f"precision: {self.avg}"

    def update(self, preds, labels, n, **kwargs):
        self.val = precision_score(labels, preds)
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count


class RecallMeter(AverageMeter):
    def __init__(self, **kwargs):
        super().__init__(name="rc", **kwargs)

    def __str__(self):
        return f"recall: {self.avg}"

    def update(self, preds, labels, n, **kwargs):
        self.val = recall_score(labels, preds)
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count


class SimilarityCorrelationMeter(AverageMeter):
    """
        measures the quality of the embeddings
        using pearson and spearman rank correlations
        between the cosine similarity scores and the labels
    """
    def __init__(self, **kwargs):
        super().__init__(name="correlation", **kwargs)
        self.val_spearman = 0
        self.val_pearson = 0
        self.sum_spearman = 0
        self.sum_pearson = 0
        self.count_spearman = 0
        self.count_pearson = 0
        self.avg_spearman = 0
        self.avg_pearson = 0

    def __str__(self):
        return f"pearsonr: {self.avg_pearson}; spearmanr: {self.avg_spearman}"

    def update(self, embeddings, labels, n, **kwargs):
        assert embeddings.shape[0] == 4
        sims_1 = F.cosine_similarity(embeddings[0], embeddings[1])
        sims_2 = F.cosine_similarity(embeddings[2], embeddings[3])
        delta_sims = sims_2 - sims_1
        pearson_corr, _ = pearsonr(labels, delta_sims)
        spearman_corr, _ = spearmanr(labels, delta_sims)
        self.val_spearman = spearman_corr
        self.val_pearson = pearson_corr
        self.sum_spearman += self.val_spearman*n
        self.sum_pearson += self.val_pearson*n
        self.count_spearman += n
        self.count_pearson += n
        self.avg_spearman = self.sum_spearman / self.count_spearman
        self.avg_pearson = self.sum_pearson / self.count_pearson
        self.avg = self.avg_pearson


class SimilarityAccuracyMeter(AverageMeter):
    def __init__(self, **kwargs):
        super().__init__(name="accuracy", **kwargs)
        self.max_acc = 0
        self.best_threshold = -1
        if self.return_predictions:
            self.all_predictions = []
            self.all_labels = []

    def __str__(self):
        line = "accuracy: {:.2f} with threshold: {:.2f}".format(self.avg, self.best_threshold)
        return line

    def update(self, embeddings, labels, n, threshold=None, **kwargs):
        assert embeddings.shape[0] == 2
        scores = 1-paired_cosine_distances(embeddings[0], embeddings[1])
        rows = list(zip(scores, labels))
        rows = sorted(rows, key=lambda x: x[0], reverse=True)
        labs = [r[1] for r in rows]
        preds = [r[0] for r in rows]
        assert(len(labs)==len(preds))

        if threshold is None:
            positive_so_far = 0
            remaining_negatives = sum(labels == 0)
            for i in range(len(rows)-1):
                score, label = rows[i]
                if label == 1:
                    positive_so_far += 1
                else:
                    remaining_negatives -= 1

                acc = (positive_so_far + remaining_negatives) / len(labels)
                if acc > self.max_acc:
                    self.max_acc = acc
                    self.best_threshold = (rows[i][0] + rows[i+1][0]) / 2
            thresh_preds = [1 if p >= self.best_threshold else 0 for p in preds]

        else:
            thresh_preds = [1 if p >= threshold else 0 for p in preds]
        
        if self.return_predictions:
            self.all_predictions.extend(preds)
            self.all_labels.extend(labs)
        
        #self.max_acc, self.best_threshold = get_accuracy_and_best_threshold_from_pr_curve(preds, labels)
        
        acc = metrics.accuracy_score(labs, thresh_preds)
        self.val = acc
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count


class SimilarityAveragePrecisionMeter(AverageMeter):
    def __init__(self, **kwargs):
        super().__init__(name="ap", **kwargs)

    def __str__(self):
        return "Average precision: {:.2f}".format(self.avg)

    def update(self, embeddings, labels, n, **kwargs):
        assert embeddings.shape[0] == 2
        scores = 1-paired_cosine_distances(embeddings[0], embeddings[1])
        items = list(zip(scores, labels))
        sorted_items = sorted(items, key=lambda x: x[0], reverse=True)
        labels = [i[1] for i in items]
        scores = [i[0] for i in items]
        if self.return_predictions:
            self.all_predictions.extend(scores)
            self.all_labels.extend(labels)
        self.val = metrics.average_precision_score(labels, scores)
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count


class SimilarityF1Meter(AverageMeter):
    def __init__(self, **kwargs):
        super().__init__(name="f1", **kwargs)
        self.best_f1 = 0
        self.best_threshold = 0
        self.best_precision = 0
        self.best_recall = 0

    def __str__(self):
        return "f1: {:.2f} with threshold: {:.2f}".format(self.avg, self.best_threshold)

    def update(self, embeddings, labels, n, **kwargs):
        scores = 1-paired_cosine_distances(embeddings[0], embeddings[1])
        rows = list(zip(scores, labels))
        rows = sorted(rows, key=lambda x: x[0], reverse=True)
        
        nextract = 0
        ncorrect = 0
        total_num = sum(labels)

        for i in range(len(rows)-1):
            score, label = rows[i]
            nextract += 1
            if label == 1:
                ncorrect += 1
            if ncorrect > 0:
                precision = ncorrect / nextract
                recall = ncorrect / total_num
                f1 = 2 * precision * recall / (precision+recall)
                if f1 > self.best_f1:
                    self.best_f1 = f1
                    self.best_precision = precision
                    self.best_recall = recall
                    self.best_threshold = (rows[i][0] + rows[i+1][0]) / 2
        #self.max_acc, self.best_threshold = get_accuracy_and_best_threshold_from_pr_curve(preds, labels)
        labels = [r[1] for r in rows]
        preds = [1 if r[0] >= self.best_threshold else 0 for r in rows]
        assert(len(labels)==len(preds))
        self.val = self.best_f1
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count


class RetrievalAccuracyMeter(AverageMeter):
    def __init__(self, print_wrong_matches=True, **kwargs):
        super().__init__(name="accuracy", **kwargs)
        self.print_wrong_matches = print_wrong_matches
        self.src2tgt = 0
        self.tgt2src = 0
        self.lines = []
        self.precision = 0
        self.recall = 0
        self.f1 = 0
        

    def __str__(self):
        accuracy = "accuracy [src2tgt: {:.2f} tgt2src: {:.2f}]".format(self.src2tgt, self.tgt2src)
        f1 = "precision: {:.2f} recall: {:.2f} f1: {:.2f}".format(self.precision, self.recall, self.f1)
        self.lines.append(accuracy)
        self.lines.append(f1)
        return "\n\n".join(self.lines)

    def update(self, src_embeddings, tgt_embeddings, source_sentences, target_sentences, **kwargs):
        cos_sims = cos_sim(src_embeddings, tgt_embeddings).detach().cpu().numpy()
        correct_src2tgt = 0
        correct_tgt2src = 0

        total_num = len(cos_sims)
        n_extract = 0
        for i in range(len(cos_sims)):
            max_idx = np.argmax(cos_sims[i])
            n_extract += 1
            if i == max_idx:
                correct_src2tgt += 1
            elif self.print_wrong_matches:
                line = f"i: {i} j: {max_idx}, {'INCORRECT' if i != max_idx else 'CORRECT'}\n"
                line += f"src: {source_sentences[i]}\n"
                line += f"tgt: {target_sentences[max_idx]}\n"
                line += f"maximum score: {cos_sims[i][max_idx]}, vs. correct score: {cos_sims[i][i]}"
                self.lines.append(line)
                results = zip(range(len(cos_sims[i])), cos_sims[i])
                results = sorted(results, key=lambda x: x[1], reverse=True)
                for idx, score in results[0:5]:
                    print("\t", idx, "(Score: %.4f)" % (score), target_sentences[idx])
            """
            self.precision = correct_src2tgt / n_extract
            self.recall = correct_src2tgt / total_num
            self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)
            """
        cos_sims = cos_sims.T
        for i in range(len(cos_sims)):
            max_idx = np.argmax(cos_sims[i])
            if i == max_idx:
                correct_tgt2src += 1
        self.src2tgt = correct_src2tgt / len(cos_sims)
        self.tgt2src = correct_tgt2src / len(cos_sims)

        logging.info("Accuracy src2tgt: {:.2f}".format(self.src2tgt*100))
        logging.info("Accuracy tgt2src: {:.2f}".format(self.tgt2src*100))

        self.avg = (self.src2tgt + self.tgt2src) / 2


if __name__ == '__main__':
    metrics = [AccuracyMeter()]
    m = Metrics(*metrics)
 
