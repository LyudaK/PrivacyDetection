import os
import time

import scipy.cluster.hierarchy as hac
import numpy as np
from scipy.spatial.distance import squareform
from sklearn import preprocessing, metrics

from feature_extraction import get_features

method='single'
from sklearn.metrics.pairwise import pairwise_distances
from metrics import *

def perform_clustering(path, method,threshold):
    prefix = '/Users/lyudakopeikina/Documents/HSE_FaceRec_tf-master/facial_clustering/lfw_ytf2%s_features.npz'
    #prefix = '/Users/lyudakopeikina/Documents/HSE_FaceRec_tf-master/facial_clustering/faces/features%s.npz'
    crop_center = False
    features_file = os.path.join(path[0], prefix % (recognizer_list[recognizer_ind][1]))
    print(features_file)
    features, labels = get_features(features_file,recognizer_list[recognizer_ind][2],recognizer_list[recognizer_ind][0])
    print(len(features[0]))
    X_norm = preprocessing.normalize(features, norm='l2')

    pair_dist = pairwise_distances(X_norm)
    timer=time.time()

    clusters = clustering_results(pair_dist, method,threshold)
    timer=time.time()-timer
    print('clustering time for',method, timer)
    predictions = -np.ones(len(labels))
    for idx, cluster in enumerate(clusters):
        predictions[cluster] = idx
    idx = len(clusters)
    for i in range(len(predictions)):
        if predictions[i] == -1:
            idx += 1
            predictions[i] = idx

    num_of_classes = len(np.unique(labels))
    num_of_clusters = len(clusters)
    print('features shape:', X_norm.shape, '#classes:', num_of_classes, '#clusters:', num_of_clusters)
    return num_of_classes, num_of_clusters, labels, predictions
def clustering_results(pairwise_dist, method, threshold=1, all_indices=None):

    clusters = []
    condensed_dist_matrix = squareform(pairwise_dist, checks=False)
    hac_clustering = hac.linkage(condensed_dist_matrix, method=method)
    #Forms flat clusters so that the original
    #An array of length n. labels[i] is the flat cluster number to which original observation i belongs.
    labels = hac.fcluster(hac_clustering, threshold, 'distance')

    if all_indices is None:
        clusters = [[ind for ind, label in enumerate(labels) if label == lbl] for lbl in set(labels)]
    else:
        for lbl in set(labels):
            cluster = [ind for ind, label in enumerate(labels) if label == lbl]
            if len(cluster) > 1:
                inf_dist = 100
                dist_matrix_cluster = pairwise_dist[cluster][:, cluster]
                penalties = np.array([[inf_dist * (all_indices[i] == all_indices[j] and i != j)
                                       for j in cluster] for i in cluster])
                dist_matrix_cluster += penalties
                condensed_dist_matrix = squareform(dist_matrix_cluster)
                Z = hac.linkage(condensed_dist_matrix, method=method)
                labels_cluster = hac.fcluster(Z, inf_dist / 2, 'distance')
                clusters.extend([[cluster[ind] for ind, label in enumerate(labels_cluster) if label == l] for l in
                                     set(labels_cluster)])
            else:
                clusters.append(cluster)

    clusters.sort(key=len, reverse=True)
    #print('clusters',clusters)
    return clusters
def eval_stats(path,method,threshold):
    num_of_classes,num_of_clusters,y_true, y_pred=perform_clustering(path,method,threshold)
    ARI=ari(y_true, y_pred)
    AMI=ami(y_true, y_pred)
    homogeneity,completeness=homogeneity_completeness(y_true, y_pred)
    precision,recall,fscore=bcubed(y_true, y_pred)
    print('fs',fscore)
    return num_of_classes,num_of_clusters,ARI,AMI,homogeneity,completeness,precision,recall,fscore

def get_avg_stats(path,method,threshold):
    counter = 3
    stats_metrics = ['#classes', '#clusters', 'ARI', 'AMI', 'homogeneity', 'completeness',
                       'precision', 'recall', 'BCubedFScore']
    stats = np.zeros((counter, len(stats_metrics)))
    for i in range(counter):
        stats[i] = eval_stats(path,method,threshold)
    mean = np.mean(stats, axis=0)
    std = np.std(stats, axis=0)
    for i, stat in enumerate(stats_metrics):
        print('%s:%.3f(%.3f) ' % (stat, mean[i], std[i]), end='')
    print('\n')


def best_threshold(path, method, cnt=20):
    #print('db_p',path)
    best_stat, prev_stat,best_threshold = 0, 0, 0
    cnt = len(path)

    for thresh in np.linspace(0.7, 1.3, 71):
        tmp = 0

        #print(thresh)
        for i, dir in enumerate(path[:cnt]):
            #print('dir',dir)
            num_of_classes, num_of_clusters, y_true, y_pred = perform_clustering(dir, method,
                                                                                         thresh)
            prec, recall, fmeasure = bcubed(y_true, y_pred)
            tmp += fmeasure

        tmp /= cnt
        if tmp > best_stat:
            best_stat = tmp
            best_threshold = thresh
        if tmp < prev_stat - 0.01:
            break
        if tmp > 0.95:
            break
        prev_stat = tmp

    print('method:', method, 'threshold:', best_threshold, 'bestStatistic:', best_stat)
    get_avg_stats(path, method, best_threshold)


recognizer_list=[['mobnet','_mobnet',0],
                 ['vgg16','_vgg16',1],
                 ['resnet50','_resnet50',1],
                 ['insightface','_insightface',0],
                 ['facenet','_facenet',0]]
recognizer_ind=0
if __name__ == '__main__':
    path_to_album=[]
    if True:
        path_to_album.append('/Users/lyudakopeikina/Desktop/faces')

    # vgg method_threshold_list = [['single', 0.98], ['average', 1.16], ['complete', 1.25], ['weighted', 1.17],['centroid', 0.78], ['median', 0.85]]
    #mobnet
    #method_threshold_list = [['single', 0.76], ['average', 0.88], ['complete', 0.98], ['weighted', 0.90],
     #                       ['centroid', 0.7], ['median', 0.7]]
    # vgg2 method_threshold_list = [['single', 0.82], ['average', 0.88], ['complete', 1.09], ['weighted', 0.93], ['centroid', 0.71], ['median', 0.75]]
    # insight method_threshold_list = [['single', 0.94], ['average', 1.1], ['complete', 1.27], ['weighted', 1.11],['centroid', 0.79], ['median', 0.82]]
   # facenet method_threshold_list = [['single', 0.73], ['average', 0.85], ['complete', 1.11], ['weighted', 0.88], ['centroid', 0.73], ['median', 0.73]]
    #method_threshold_list = [['single', 0.76], ['average', 0.95], ['complete', 1.21], ['weighted', 1.06],
    #                         ['centroid', 0.71], ['median', 0.7]]

    method_threshold_list = [['single', 0.76], ['average', 0.88], ['complete', 0.98], ['weighted', 0.90],
                             ['centroid', 0.7], ['median', 0.7]]

    for method, thresh in method_threshold_list:
        eval_stats(path_to_album, method,thresh)

    


