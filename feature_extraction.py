from __future__ import absolute_import

import os.path
import os
from sklearn import preprocessing

from keras.engine import Model
from keras.preprocessing import image
from keras import backend as K
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

import numpy as np

from tf_inference import TensorFlowInference

np.random.seed(123)

KERAS=1
TF=0

IMAGES = '/Users/lyudakopeikina/Desktop/faces'  # _faces'
crop_center = False

def extract_keras_features(model, img_filepath, crop_center):
    _, w, h, _ = model.input.shape
    w, h = int(w), int(h)
    if crop_center:
        orig_w, orig_h = 250, 250
        img = image.load_img(img_filepath, target_size=(orig_w, orig_h))
        w1, h1 = 128, 128
        dw = (orig_w - w1) / 2
        dh = (orig_h - h1) / 2
        box = (dw, dh, orig_w - dw, orig_h - dh)
        img = img.crop(box)
        img = img.resize((w, h))
    else:
        img = image.load_img(img_filepath, target_size=(w, h))  # (224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x).reshape(-1)
    return preds

def load_recognizer(name):
    print('name',name)
    if name is 'mobnet':
        tfInference=TensorFlowInference('/Users/lyudakopeikina/Documents/models/age_gender_tf2_new-01-0.14-0.92.pb',input_tensor='input_1:0',output_tensor='global_pooling/Mean:0',convert2BGR=True, imageNetUtilsMean=True)
    elif name is 'facenet':
        tfInference = TensorFlowInference('/Users/lyudakopeikina/Documents/models/20180402-114759.pb',
                                          input_tensor='input:0', output_tensor='embeddings:0',
                                          learning_phase_tensor='phase_train:0',
                                          convert2BGR=False)  # embeddings, InceptionResnetV1/Repeat_2/block8_5/Relu, InceptionResnetV1/Repeat_1/block17_10/Relu
    elif name is 'insightface':
        tfInference=TensorFlowInference('/Users/lyudakopeikina/Documents/models/insightface.pb',input_tensor='img_inputs:0',output_tensor='resnet_v1_50/E_BN2/Identity:0',learning_phase_tensor='dropout_rate:0',convert2BGR=False,additional_input_value=0.9)
    else:
        raise ValueError('Invalid name')
    return tfInference
def load_recognizer_keras(name):

    K.set_learning_phase(0)
    if name == 'vgg16':
        model_name, layer = 'vgg16', 'fc7/relu'
    elif name == 'resnet50':
        model_name, layer = 'resnet50', 'avg_pool'
    model = VGGFace(model=model_name)

    out = model.get_layer(layer).output
    cnn_model = Model(model.input, out)
    cnn_model.summary()
    return cnn_model
img_extensions = ['.jpg', '.jpeg', '.png']

def is_image(path):
    _, file_extension = os.path.splitext(path)
    return file_extension.lower() in img_extensions
def get_files(dir):
    return [[d, os.path.join(d, f)] for d in next(os.walk(dir))[1] for f in next(os.walk(os.path.join(dir, d)))[2]
            if not f.startswith(".") and is_image(f)]



def extract_per_img(face_img,model,use_framework):
    if use_framework == 1:
        return np.array(extract_keras_features(model, face_img, crop_center) )
    elif use_framework == 0:
        return np.array(model.extract_features(face_img, crop_center=crop_center))

def get_features(features_file,use_framework,recognizer):
    if not os.path.exists(features_file):
        dirs_and_files = np.array(get_files(IMAGES))
        dirs = dirs_and_files[:, 0]
        files = dirs_and_files[:, 1]
        print(dirs)

        label_enc = preprocessing.LabelEncoder()
        label_enc.fit(dirs)
        y = label_enc.transform(dirs)

        if use_framework == KERAS:
            cnn_model = load_recognizer_keras(recognizer)
            cnn_model.summary()
            X = np.array(
                [extract_keras_features(cnn_model, os.path.join(IMAGES, f), crop_center) for f in
                 files])
        elif use_framework==TF:
            tfInference = load_recognizer(recognizer)
            X = np.array(
                [tfInference.extract_features(os.path.join(IMAGES, filepath), crop_center=crop_center) for
                 filepath in files])
            tfInference.close_session()
            print(features_file)
            np.savez(features_file, x=X, y=y)
            return X,y
    else:
        filenpz = np.load(features_file)
        X = filenpz['x']
        y = filenpz['y']
        label_enc = preprocessing.LabelEncoder()
        label_enc.fit(y)
        y = label_enc.transform(y)
        return X,y


#recognizer_list=[['mobnet','_mobnet'],['vgg16','_vgg16'],['resnet50','_resnet50'],['insightface','_insightface'],['facenet','_facenet']]
#recognizer_ind=1
#use_framework=KERAS
if __name__ == '__main__':

    '''
    prefix='lfw_ytf2_features%s.npz'
    crop_center = False
    features_file=os.path.join(IMAGES,prefix%(recognizer_list[recognizer_ind][1]))
    print(features_file)
    features,labels=get_features(features_file,use_framework,recognizer_list[recognizer_ind][0])
    '''
    # features_file = 'lfw_ytf2_mobnet_features.npz'
    # features_file = 'lfw_ytf2_facenet_features.npz'
    # features_file = 'lfw_ytf2_insightface_features.npz'
    # features_file = 'lfw_ytf2_vgg16_features.npz'
    # features_file = 'lfw_ytf2_resnet50_features.npz'


    #np.savez(features_file, x=features, y=labels)

