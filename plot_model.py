import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import CustomObjectScope
from train import iou


if __name__ == '__main__':
    with CustomObjectScope({'iou':iou}):
        unet = tf.keras.models.load_model(r'files\model.h5')
        unetpp = tf.keras.models.load_model(r'files\unetpp.h5')
    plot_model(unet, to_file='unet.png', show_shapes=True)
    plot_model(unetpp, to_file='unetpp.png', show_shapes=True)