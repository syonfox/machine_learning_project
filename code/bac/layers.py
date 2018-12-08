'''
Custom Keras layers used on the pastiche model.
'''


from keras import backend as K
from keras.engine.topology import Layer
from keras.layers.merge import add
from keras.engine import InputSpec
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization, InstanceNormalization
from keras.layers.convolutional import Deconvolution2D,  Conv2D,UpSampling2D,Cropping2D
from VGG16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers.advanced_activations import LeakyReLU
from keras.applications.imagenet_utils import  preprocess_input

import numpy as np
import tensorflow as tf


class InputNormalize(Layer):
    def __init__(self, **kwargs):
        super(InputNormalize, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def compute_output_shape(self,input_shape):
        return input_shape

    def call(self, x, mask=None):
        #x = (x - 127.5)/ 127.5
        return x/255.



class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def get_output_shape_for(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')


#
# class InstanceNormalization(Layer):
#     def __init__(self, **kwargs):
#         super(InstanceNormalization, self).__init__(**kwargs)
#         self.epsilon = 1e-3
#
#
#     def call(self, x, mask=None):
#         mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
#         return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, self.epsilon)))
#
#
#     def compute_output_shape(self,input_shape):
#         return input_shape

# class InstanceNormalization(Layer):
#     def __init__(self, epsilon=1e-5, weights=None,
#                  beta_init='zero', gamma_init='one', **kwargs):
#         self.beta_init = initializers.get(beta_init)
#         self.gamma_init = initializers.get(gamma_init)
#         self.epsilon = epsilon
#         super(InstanceNormalization, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         # This currently only works for 4D inputs: assuming (B, H, W, C)
#         self.input_spec = [InputSpec(shape=input_shape)]
#         shape = (1, 1, 1, input_shape[-1])
#
#         self.gamma = self.gamma_init(shape, name='{}_gamma'.format(self.name))
#         self.beta = self.beta_init(shape, name='{}_beta'.format(self.name))
#         self.trainable_weights = [self.gamma, self.beta]
#
#         self.built = True
#
#     def call(self, x, mask=None):
#         # Do not regularize batch axis
#         reduction_axes = [1, 2]
#
#         mean, var = tf.nn.moments(x, reduction_axes,
#                                   shift=None, name=None, keep_dims=True)
#         x_normed = tf.nn.batch_normalization(x, mean, var, self.beta, self.gamma, self.epsilon)
#         return x_normed
#
#     def get_config(self):
#         config = {"epsilon": self.epsilon}
#         base_config = super(InstanceNormalization, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
#
#
# class ConditionalInstanceNormalization(InstanceNormalization):
#     def __init__(self, targets, nb_classes, **kwargs):
#         self.targets = targets
#         self.nb_classes = nb_classes
#         super(ConditionalInstanceNormalization, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         # This currently only works for 4D inputs: assuming (B, H, W, C)
#         self.input_spec = [InputSpec(shape=input_shape)]
#         shape = (self.nb_classes, 1, 1, input_shape[-1])
#
#         self.gamma = self.gamma_init(shape, name='{}_gamma'.format(self.name))
#         self.beta = self.beta_init(shape, name='{}_beta'.format(self.name))
#         self.trainable_weights = [self.gamma, self.beta]
#
#         self.built = True
#
#     def call(self, x, mask=None):
#         # Do not regularize batch axis
#         reduction_axes = [1, 2]
#
#         mean, var = tf.nn.moments(x, reduction_axes,
#                                   shift=None, name=None, keep_dims=True)
#
#         # Get the appropriate lines of gamma and beta
#         beta = tf.gather(self.beta, self.targets)
#         gamma = tf.gather(self.gamma, self.targets)
#         x_normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, self.epsilon)
#
#         return x_normed
