from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time, datetime
import cv2
import shutil
import pickle

import numpy as np
from configparser import ConfigParser

from facial_analysis import FacialImageProcessing, is_image



from sklearn import preprocessing

from agl_clustering import clustering_results
from feature_extraction import extract_per_img

from feature_extraction import load_recognizer, load_recognizer_keras

recognizer_list=[['mobnet','_mobnet',0],
                 ['vgg16','_vgg16',1],
                 ['resnet50','_resnet50',1],
                 ['insightface','_insightface',0],
                 ['facenet','_facenet',0]]
recognizer_ind=4
groupSize = 2
distanceThreshold = 1.35

img_size = 224
if recognizer_list[recognizer_ind][2] == 1:
    model = load_recognizer_keras(recognizer_list[recognizer_ind][0])
elif recognizer_list[recognizer_ind][2] == 0:
    model = load_recognizer(recognizer_list[recognizer_ind][0])
def process_image(imgProcessing, img):


    def process(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = time.time()
        bounding_boxes, points = imgProcessing.detect_faces(img)
        elapsed = time.time() - t
        facial_features, bboxes = [], []
        for b in bounding_boxes:
            b = [int(bi) for bi in b]
            # print(b,img.shape)
            x1, y1, x2, y2 = b[0:4]
            if x2 > x1 and y2 > y1:
                img_h, img_w, _ = img.shape
                w, h = x2 - x1, y2 - y1
                dw, dh = 10, 10  # max(w//8,10),max(h//8,10) #w//6,h//6
                # sz=max(w+2*dw,h+2*dh)
                # dw,dh=(sz-w)//2,(sz-h)//2
                x1, x2 = x1 - dw, x2 + dw
                y1, y2 = y1 - dh, y2 + dh

                boxes = [[x1, y1, x2, y2]]

                if False:  # oversampling
                    delta = 10
                    boxes.append([x1 - delta, y1 - delta, x2 - delta, y2 - delta])
                    boxes.append([x1 - delta, y1 + delta, x2 - delta, y2 + delta])
                    boxes.append([x1 + delta, y1 - delta, x2 + delta, y2 - delta])
                    boxes.append([x1 + delta, y1 + delta, x2 + delta, y2 + delta])

                for ind in range(len(boxes)):
                    if boxes[ind][0] < 0:
                        boxes[ind][0] = 0
                    if boxes[ind][2] > img_w:
                        boxes[ind][2] = img_w
                    if boxes[ind][1] < 0:
                        boxes[ind][1] = 0
                    if boxes[ind][3] > img_h:
                        boxes[ind][3] = img_h

                for (x1, y1, x2, y2) in boxes[::-1]:
                    face_img = img[y1:y2, x1:x2, :]
                    cv2.imwrite('face.jpg', cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                    face_img = 'face.jpg'
                    t = time.time()
                    features = extract_per_img(face_img,model,recognizer_list[recognizer_ind][2])

                facial_features.append(features)
                bboxes.append(boxes[0])
        # print('facial_features', facial_features)
        return bboxes, points, facial_features

    # print(model_desc[model_ind][0])

    height, width, channels = img.shape
    bounding_boxes, _, facial_features = process(img)
    facial_images = []
    has_center_face = False
    for bb in bounding_boxes:
        x1, y1, x2, y2 = bb[0:4]
        face_img = cv2.resize(img[y1:y2, x1:x2, :], (img_size, img_size))
        facial_images.append(face_img)

    return facial_images, facial_features, has_center_face


def perform_splitting(all_indices, all_features, no_images_in_cluster):
    def feature_distance(i, j):
        dist = np.sqrt(np.sum((all_features[i] - all_features[j]) ** 2))
        return [dist]

    num_faces = len(all_indices)
    if num_faces < no_images_in_cluster:
        return []

    t = time.time()
    pair_dist = np.array([[feature_distance(i, j) for j in range(num_faces)] for i in range(num_faces)])
    dist_matrix = np.clip(np.sum(pair_dist, axis=2), a_min=0, a_max=None)
    print('all_indices', all_indices)
    clusters = clustering_results(dist_matrix, 'weighted',distanceThreshold, all_indices)
    elapsed = time.time() - t
    # print('clustering elapsed=%f'%(elapsed))

    print('clusters', clusters)
    print('len', len(clusters))

    def is_good_cluster(cluster):
        res = len(cluster) >= no_images_in_cluster
        return res

    def private_indices(all_indices, filtered_clusters):
        file_face = []
        c = 0
        private_photos = []
        private_faces = []
        for file in (all_indices):
            file_face.append([file, c])
            c = c + 1
        # print('file_face', file_face)
        for ff in file_face:
            for fc in filtered_clusters:
                if (ff[1] in fc):
                    private_photos.append(ff[0])
        return private_photos, file_face

    filtered_clusters = [cluster for cluster in clusters if is_good_cluster(cluster)]
    print('filtered_clusters', filtered_clusters)
    print('len', len(filtered_clusters))
    private_photos, file_face = private_indices(all_indices, filtered_clusters)
    print('private_photos', private_photos)

    return set(private_photos), file_face, clusters


def split_data(imgProcessing, album_dir, groupSize):
    features_file = os.path.join(album_dir, 'test%s.dump' % recognizer_list[recognizer_ind][1])
    t = time.time()
    if os.path.exists(features_file):
        with open(features_file, "rb") as f:
            files = pickle.load(f)
            all_facial_images = pickle.load(f)
            all_features = pickle.load(f)
            all_indices = pickle.load(f)
            private_photo_indices = pickle.load(f)
            y_true = pickle.load(f)
    else:
        dirs_and_files = np.array([[d, os.path.join(d, f)] for d in next(os.walk(album_dir))[1] for f
                                   in next(os.walk(os.path.join(album_dir, d)))[2] if is_image(f)])

        dirs = dirs_and_files[:, 0]
        files = dirs_and_files[:, 1]
        # print(files)
        # print('opened')
        print(dirs)
        print(len(np.unique(dirs)))
        print('files', files)

        label_enc = preprocessing.LabelEncoder()
        label_enc.fit(dirs)
        y_true = label_enc.transform(dirs)
        print('y=', y_true)
        # files = [f for f in next(os.walk(album_dir))[2] if is_image(f)]
        # files=files[:20]
        all_facial_images, all_features, all_indices, private_photo_indices = [], [], [], []
        for i, fpath in enumerate(files):
            print(fpath)
            full_photo = cv2.imread(os.path.join(album_dir, fpath))
            facial_images, facial_features, has_center_face = process_image(imgProcessing, full_photo)
            if len(facial_images) == 0:
                full_photo_t = cv2.transpose(full_photo)
                rotate90 = cv2.flip(full_photo_t, 1)
                facial_images, facial_features, has_center_face = process_image(imgProcessing, rotate90)
                if len(facial_images) == 0:
                    rotate270 = cv2.flip(full_photo_t, 0)
                    facial_images, facial_features, has_center_face = process_image(imgProcessing,
                                                                                    rotate270)
            if has_center_face:
                # private_photo_indices.append(i)
                print('hello ind')
            all_facial_images.extend(facial_images)
            for features in facial_features:
                features = features / np.sqrt(np.sum(features ** 2))
                all_features.append(features)
            all_indices.extend([i] * len(facial_images))
            # print('all_features',all_features)

            print('Processed photos: %d/%d\r' % (i + 1, len(files)), end='')
            sys.stdout.flush()

        with open(features_file, "wb") as f:
            pickle.dump(files, f)
            pickle.dump(all_facial_images, f)
            pickle.dump(all_features, f)
            pickle.dump(all_indices, f)
            pickle.dump(private_photo_indices, f)
            pickle.dump(y_true, f)
        print('features dumped into', features_file)

    all_features = np.array(all_features)
    print(np.shape(all_features))

    private_photo_indices, file_face, all_clusters = perform_splitting(all_indices, all_features, groupSize)
    print(private_photo_indices)
    # files = [f for f in next(os.walk(album_dir))[2] if is_image(f)]
    print('len(files)', len(files))
    y_pred = np.ones(len(files))
    for i in private_photo_indices:
        y_pred[i] = 0
    '''    
    print('y_true', y_true)
    print('y_pred', y_pred)
    print('Accuracy:', accuracy_score(y_true, y_pred))
    print('F1 score:', f1_score(y_true, y_pred))
    print('Recall:', recall_score(y_true, y_pred))
    print('Precision:', precision_score(y_true, y_pred))
    print('\n clasification report:\n', classification_report(y_true, y_pred))
    print('\n confussion matrix:\n', confusion_matrix(y_true, y_pred))
    false_negative=[i for i in range(len(files)//2) if i not in private_photo_indices]
    print(false_negative)
    print(file_face)
    fn_faces=[ff[1] for ff in file_face if ff[0] in false_negative]
    print(fn_faces)
    '''

    if True:
        res_dir = os.path.join(album_dir, 'clusters')
        if os.path.exists(res_dir):
            shutil.rmtree(res_dir, ignore_errors=True)
            time.sleep(2)

        clust_dir = os.path.join(res_dir, 'private')
        os.makedirs(clust_dir)
        for ind in private_photo_indices:
            full_photo = cv2.imread(os.path.join(album_dir, files[ind]))
            r = 200.0 / full_photo.shape[1]
            dim = (200, int(full_photo.shape[0] * r))
            full_photo = cv2.resize(full_photo, dim)
            cv2.imwrite(os.path.join(clust_dir, '%s.jpg' % (files[ind].split("/")[1])), full_photo)
        clust_dir = os.path.join(res_dir, 'public')
        os.makedirs(clust_dir)
        idx = list(range(0, len(files)))
        public_photo_indices = [i for i in idx if i not in private_photo_indices]
        for ind in public_photo_indices:
            full_photo = cv2.imread(os.path.join(album_dir, files[ind]))
            r = 200.0 / full_photo.shape[1]
            dim = (200, int(full_photo.shape[0] * r))
            full_photo = cv2.resize(full_photo, dim)
            cv2.imwrite(os.path.join(clust_dir, '%s.jpg' % (files[ind].split("/")[1])), full_photo)




if __name__ == '__main__':
    config = ConfigParser()
    config.read('config.txt')
    default_config = config['DEFAULT']
    # print(default_config['InputDirectory'])

    imgProcessing = FacialImageProcessing(print_stat=False, minsize=112)
    # mobnet 1.1
    # resnet 1.1
    # vgg16 1.25
    # facenet 1.35

    split_data(imgProcessing, '/Users/lyudakopeikina/Downloads/105APPLE', groupSize)
    imgProcessing.close()

