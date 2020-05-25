# Clustering example using LFW data:
import os
from datetime import time

import pandas as pd
from matplotlib import pyplot as plt
import argparse
import json
import scipy.io as sio
from sklearn import metrics, preprocessing

from clustering import cluster
import numpy as np
np.random.seed(42)

def approximate_rank_order_clustering(vectors,threshold):
    clusters = cluster(vectors, n_neighbors=130, thresh=threshold)

    return clusters
def plot_histogram(lfw_dir):
    filecount_dict = {}
    for root, dirs, files in os.walk(lfw_dir):
        for dirname in dirs:
            n_photos = len(os.listdir(os.path.join(root, dirname)))
            print(os.listdir(os.path.join(root, dirname)))
            filecount_dict[dirname] = n_photos
            print(dirname)
    print("No of unique people: {}".format(len(filecount_dict.keys())))
    df = pd.DataFrame(filecount_dict.items(), columns=['Name', 'Count'])
    print("Singletons : {}\nTwo :{}\n".format((df['Count'] == 1).sum(),
                                              (df['Count'] == 2).sum()))
    plt.hist(df['Count'], bins=max(df['Count']))
    plt.title('Cluster Sizes')
    plt.xlabel('No of images in folder')
    plt.ylabel('No of folders')
    plt.show()
def fscore(p_val, r_val, beta=1.0):
    """Computes the F_{beta}-score of given precision and recall values."""
    return (1.0 + beta ** 2) * (p_val * r_val / (beta ** 2 * p_val + r_val))


def mult_precision(el1, el2, cdict, ldict):
    """Computes the multiplicity precision for two elements."""
    return min(len(cdict[el1] & cdict[el2]), len(ldict[el1] & ldict[el2])) \
           / float(len(cdict[el1] & cdict[el2]))


def mult_recall(el1, el2, cdict, ldict):
    return min(len(cdict[el1] & cdict[el2]), len(ldict[el1] & ldict[el2])) \
           / float(len(ldict[el1] & ldict[el2]))


def precision(cdict, ldict):
    return np.mean([np.mean([mult_precision(el1, el2, cdict, ldict) \
                             for el2 in cdict if cdict[el1] & cdict[el2]]) for el1 in cdict])


def recall(cdict, ldict):
    """Computes overall extended BCubed recall for the C and L dicts."""
    return np.mean([np.mean([mult_recall(el1, el2, cdict, ldict) \
                             for el2 in cdict if ldict[el1] & ldict[el2]]) for el1 in cdict])


def get_BCubed_set(y_vals):
    dic = {}
    for i, y in enumerate(y_vals):
        dic[i] = set([y])
    return dic


def BCubed_stat(y_true, y_pred, beta=1.0):
    cdict = get_BCubed_set(y_true)
    ldict = get_BCubed_set(y_pred)
    p = precision(cdict, ldict)
    r = recall(cdict, ldict)
    f = fscore(p, r, beta)
    return (p, r, f)
def evaluate_clustering(nclasses,nclusters,clusters,labels):
    ari=metrics.adjusted_rand_score(labels, clusters)
    ami=metrics.adjusted_mutual_info_score(labels, clusters,average_method ='arithmetic')
    homogeneity,completeness,v_measure=metrics.homogeneity_completeness_v_measure(labels, clusters)
    bcubed_precision,bcubed_recall,bcubed_fmeasure=BCubed_stat(labels, clusters)
    return nclasses,nclusters,ari,ami,homogeneity,completeness,v_measure,bcubed_precision,bcubed_recall,bcubed_fmeasure
def get_avg_stat(vectors,labels,threshold):
    counter = 5 #len(np.unique(labels))-590
    stats_metrics = ['classes', 'clusters', 'ARI', 'AMI', 'homogeneity', 'completeness', 'v-measure',
                       'BCubed_precision', 'BCubed_recall', 'BCubed_FMeasure']
    stats = np.zeros((counter, len(stats_metrics)))
    for i in range(counter):
        stats[i] = perform_clustering(vectors,threshold)
    mean = np.mean(stats, axis=0)
    std = np.std(stats, axis=0)
    for i, stat in enumerate(stats_metrics):
        print('%s:%.3f(%.3f) ' % (stat, mean[i], std[i]), end='')
    print('\n')
def perform_clustering(vectors,threshold):
    clusters_thresholds = approximate_rank_order_clustering(vectors,threshold)
    for clusters in clusters_thresholds:

        y_pred = -np.ones(len(labels))

        for ind, cluster in enumerate(clusters['clusters']):
            y_pred[list(cluster)] = ind
        ind = len(clusters)
        for i in range(len(y_pred)):
            if y_pred[i] == -1:
                ind += 1
                y_pred[i] = ind
            # print(labels)
            # print(y_pred)
        nclasses, nclusters, ari, ami, homogeneity, completeness, v_measure, bcubed_precision, bcubed_recall, bcubed_fmeasure = \
            evaluate_clustering(595, len(clusters['clusters']), labels, y_pred)
        # get_avg_stat(32,len(clusters['clusters']),labels,y_pred)

    return nclasses,nclusters,ari,ami,homogeneity,completeness,v_measure,bcubed_precision,bcubed_recall,bcubed_fmeasure

if __name__ == '__main__':
    # plot_histogram(args['lfw_path'])
    #features_file='/Users/lyudakopeikina/Documents/HSE_FaceRec_tf-master/age_gender_identity/faces/features_vgg16.npz'
    #features_file = '/Users/lyudakopeikina/Downloads/lfw/lfw_features_vgg16.npz'
    #features_file = '/Users/lyudakopeikina/Documents/HSE_FaceRec_tf-master/age_gender_identity/lfw_ytf_facenet_features.npz'
    features_file = '/Users/lyudakopeikina/Documents/HSE_FaceRec_tf-master/facial_clustering/lfw_ytf2_resnet50_features.npz'
    f = np.load(features_file)
    vectors = f['x']
    vectors = preprocessing.normalize(vectors, norm='l2')
    #labels = f['y_true']
    labels= f['y']

    label_enc = preprocessing.LabelEncoder()
    label_enc.fit(labels)
    labels = label_enc.transform(labels)
    print('labels',labels[0:100])

    clusters_thresholds = approximate_rank_order_clustering(vectors)

        #clusters_at_th = clusters_thresholds[0]
        #np.save('clusters_aro_resnet50.npy',clusters_at_th)

    for clusters in clusters_thresholds:

        print("No of clusters: {}".format(len(clusters['clusters'])))
        print("Threshold : {}".format(clusters['threshold']))
        y_pred = -np.ones(len(labels))
        for ind, cluster in enumerate(clusters['clusters']):
            y_pred[list(cluster)] = ind
        ind = len(clusters)
        for i in range(len(y_pred)):
            if y_pred[i] == -1:
                ind += 1
                y_pred[i] = ind
        evaluate_clustering(595,len(clusters['clusters']),labels,y_pred)




