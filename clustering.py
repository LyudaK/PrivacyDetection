
import pyflann
import numpy as np
from time import time
from profilehooks import profile
from multiprocessing import Pool
from functools import partial
import json

def build_index(dataset, n_neighbors):
    pyflann.set_distance_type(distance_type='euclidean')
    flann = pyflann.FLANN()
    params = flann.build_index(dataset,
                               algorithm='kdtree',
                               trees=4
                               )
    nearest_neighbors, dists = flann.nn_index(dataset, n_neighbors,
                                              checks=params['checks'])

    return nearest_neighbors, dists


def create_neighbor_lookup(nearest_neighbors):
    nn_lookup = {}
    for i in range(nearest_neighbors.shape[0]):
        nn_lookup[i] = nearest_neighbors[i, :]
    return nn_lookup

def calculate_symmetric_dist_row(nearest_neighbors, nn_lookup, row_no):

    dist_row = np.zeros([1, nearest_neighbors.shape[1]])
    f1 = nn_lookup[row_no]
    # print "f1 : ", f1
    for idx, neighbor in enumerate(f1[1:]):
        Oi = idx+1
        co_neighbor = True
        try:
            row = nn_lookup[neighbor]
            Oj = np.where(row == row_no)[0][0] + 1
            # print 'Correct Oj: {}'.format(Oj)
        except IndexError:
            Oj = nearest_neighbors.shape[1]+1
            co_neighbor = False

        #dij
        f11 = set(f1[0:Oi])
        f21 = set(nn_lookup[neighbor])
        dij = len(f11.difference(f21))
        #dji
        f12 = set(f1)
        f22 = set(nn_lookup[neighbor][0:Oj])
        dji = len(f22.difference(f12))

        if not co_neighbor:
            dist_row[0, Oi] = 9999.0
        else:
            dist_row[0, Oi] = float(dij + dji)/min(Oi, Oj)

    return dist_row


def calculate_symmetric_dist(app_nearest_neighbors):

    dist_calc_time = time()
    nn_lookup = create_neighbor_lookup(app_nearest_neighbors)
    d = np.zeros(app_nearest_neighbors.shape)
    p = Pool(processes=4)
    func = partial(calculate_symmetric_dist_row, app_nearest_neighbors, nn_lookup)
    results = p.map(func, range(app_nearest_neighbors.shape[0]))
    for row_no, row_val in enumerate(results):
        d[row_no, :] = row_val
    d_time = time()-dist_calc_time
    print("Distance calculation time : {}".format(d_time))
    return d


def aro_clustering(app_nearest_neighbors, distance_matrix, thresh):

    clusters = []

    nodes = set(list(np.arange(0, distance_matrix.shape[0])))
    # print 'Nodes initial : {}'.format(nodes)
    tc = time()
    plausible_neighbors = create_plausible_neighbor_lookup(
                                                            app_nearest_neighbors,
                                                            distance_matrix,
                                                            thresh)
    # print 'Time to create plausible_neighbors lookup : {}'.format(time()-tc)
    ctime = time()
    while nodes:
        # Get a node :
        n = nodes.pop()

        # This contains the set of connected nodes :
        group = {n}

        # Build a queue with this node in it :
        queue = [n]

        # Iterate over the queue :
        while queue:
            n = queue.pop(0)
            neighbors = plausible_neighbors[n]
            # Remove neighbors we've already visited :
            neighbors = nodes.intersection(neighbors)
            neighbors.difference_update(group)

            # Remove nodes from the global set :
            nodes.difference_update(neighbors)

            # Add the connected neighbors :
            group.update(neighbors)

            # Add the neighbors to the queue to visit them next :
            queue.extend(neighbors)
        # Add the group to the list of groups :
        clusters.append(group)

    # print 'Clustering Time : {}'.format(time()-ctime)
    return clusters


def create_plausible_neighbor_lookup(app_nearest_neighbors,
                                     distance_matrix,
                                     thresh):
    n_vectors = app_nearest_neighbors.shape[0]
    plausible_neighbors = {}
    for i in range(n_vectors):
        plausible_neighbors[i] = set(list(app_nearest_neighbors[i,
                                     np.where(
                                            distance_matrix[i, :] <= thresh)]
                                             [0]))
    return plausible_neighbors


def cluster(descriptor_matrix, n_neighbors=10, thresh=[2]):
    app_nearest_neighbors, dists = build_index(descriptor_matrix, n_neighbors)
    distance_matrix = calculate_symmetric_dist(app_nearest_neighbors)
    clusters = []
    for th in thresh:
        clusters_th = aro_clustering(app_nearest_neighbors, distance_matrix, th)
        print("N Clusters: {}, thresh: {}".format(len(clusters_th), th))
        clusters.append({'clusters': clusters_th, 'threshold': th})
    return clusters


if __name__ == '__main__':
    descriptor_matrix = np.random.rand(20, 10)
    app_nearest_neighbors, dists = build_index(descriptor_matrix, n_neighbors=2)
    distance_matrix = calculate_symmetric_dist(app_nearest_neighbors)
    timer = time.time()
    clusters = cluster(descriptor_matrix, n_neighbors=2)
    print('clustering time', timer)
    clusters_to_be_saved = {}
    for i, cluster in enumerate(clusters[0]["clusters"]):
        c = [int(x) for x in list(cluster)]
        clusters_to_be_saved[i] = c
    with open("clusters.json", "w") as f:
        json.dump(clusters_to_be_saved, f)


