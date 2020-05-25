import tensorflow as tf
import numpy as np
from scipy import misc

class TensorFlowInference:
    def __init__(self, frozen_graph_filename, input_tensor, output_tensor, learning_phase_tensor=None, convert2BGR=True,
                 imageNetUtilsMean=True, additional_input_value=0):
        graph = load_graph(frozen_graph_filename, '')
        print([n.name for n in graph.as_graph_def().node if 'input' in n.name])

        graph_op_list = list(graph.get_operations())
        print([n.name for n in graph_op_list if 'keras_learning' in n.name])

        self.tf_sess = tf.Session(graph=graph)

        self.tf_input_image = graph.get_tensor_by_name(input_tensor)
        print('tf_input_image=', self.tf_input_image)
        self.tf_output_features = graph.get_tensor_by_name(output_tensor)
        print('tf_output_features=', self.tf_output_features)
        self.tf_learning_phase = graph.get_tensor_by_name(learning_phase_tensor) if learning_phase_tensor else None;
        print('tf_learning_phase=', self.tf_learning_phase)
        if self.tf_input_image.shape.dims is None:
            w = h = 160
        else:
            _, w, h, _ = self.tf_input_image.shape
        self.w, self.h = int(w), int(h)
        print('input w,h', self.w, self.h, ' output shape:', self.tf_output_features.shape)

        self.convert2BGR = convert2BGR
        self.imageNetUtilsMean = imageNetUtilsMean
        self.additional_input_value = additional_input_value

    def preprocess_image(self, img_filepath, crop_center):
        if crop_center:
            orig_w, orig_h = 250, 250
            img = misc.imread(img_filepath, mode='RGB')
            img = misc.imresize(img, (orig_w, orig_h), interp='bilinear')
            w1, h1 = 128, 128
            dw = (orig_w - w1) // 2
            dh = (orig_h - h1) // 2
            box = (dw, dh, orig_w - dw, orig_h - dh)
            img = img[dh:-dh, dw:-dw]
        else:
            img = misc.imread(img_filepath, mode='RGB')

        x = misc.imresize(img, (self.w, self.h), interp='bilinear').astype(float)

        if self.convert2BGR:
            # 'RGB'->'BGR'
            x = x[..., ::-1]
            # Zero-center by mean pixel
            if self.imageNetUtilsMean:  # imagenet.utils caffe
                x[..., 0] -= 103.939
                x[..., 1] -= 116.779
                x[..., 2] -= 123.68
            else:  # vggface-2
                x[..., 0] -= 91.4953
                x[..., 1] -= 103.8827
                x[..., 2] -= 131.0912
        else:
            # x=(x-127.5)/128.0
            x /= 127.5
            x -= 1.
            # x=x/128.0-1.0
        return x

    def extract_features(self, img_filepath, crop_center=False):
        x = self.preprocess_image(img_filepath, crop_center)
        x = np.expand_dims(x, axis=0)
        feed_dict = {self.tf_input_image: x}
        if self.tf_learning_phase is not None:
            feed_dict[self.tf_learning_phase] = self.additional_input_value
        preds = self.tf_sess.run(self.tf_output_features, feed_dict=feed_dict).reshape(-1)
        # preds = self.tf_sess.run(self.tf_output_features, feed_dict=feed_dict).mean(axis=(0,1,2)).reshape(-1)
        return preds

    def close_session(self):
        self.tf_sess.close()

def load_graph(frozen_graph_filename, prefix=''):
    with tf.gfile.GFile(frozen_graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name=prefix)
    return graph