'''
This module contains functions for building the pastiche model.
'''

import keras
# from keras.models import Model
# from keras.layers import (Convolution2D, Activation, UpSampling2D,
#                           ZeroPadding2D, Input, BatchNormalization,
#                           merge, Lambda)
from layers import (ReflectionPadding2D, InstanceNormalization,
                    ConditionalInstanceNormalization)
from keras.initializations import normal

from keras.models import *
from keras.layers import *
from keras.optimizers import *

# Initialize weights with normal distribution with std 0.01
def weights_init(shape, name=None, dim_ordering=None):
    return normal(shape, scale=0.01, name=name)


def conv(x, n_filters, kernel_size=3, stride=1, relu=True, nb_classes=1, targets=None):
    '''
    Reflection padding, convolution, instance normalization and (maybe) relu.
    '''
    if not kernel_size % 2:
        raise ValueError('Expected odd kernel size.')
    pad = (kernel_size - 1) / 2
    o = ReflectionPadding2D(padding=(pad, pad))(x)
    o = Convolution2D(n_filters, kernel_size, kernel_size,
                      subsample=(stride, stride), init=weights_init)(o)
    # o = BatchNormalization()(o)
    if nb_classes > 1:
        o = ConditionalInstanceNormalization(targets, nb_classes)(o)
    else:
        o = InstanceNormalization()(o)
    if relu:
        o = Activation('relu')(o)
    return o


def residual_block(x, n_filters, nb_classes=1, targets=None):
    '''
    Residual block with 2 3x3 convolutions blocks. Last one is linear (no ReLU).
    '''
    o = conv(x, n_filters)
    # Linear activation on second conv
    o = conv(o, n_filters, relu=False, nb_classes=nb_classes, targets=targets)
    # Shortcut connection
    o = merge([o, x], mode='sum')
    return o


def upsampling(x, n_filters, nb_classes=1, targets=None):
    '''
    Upsampling block with nearest-neighbor interpolation and a conv block.
    '''
    o = UpSampling2D()(x)
    o = conv(o, n_filters, nb_classes=nb_classes, targets=targets)
    return o


def pastiche_model(img_size, width_factor=2, nb_classes=1, targets=None):
    k = width_factor
    x = Input(shape=(img_size, img_size, 3))
    o = conv(x, 16 * k, kernel_size=9, nb_classes=nb_classes, targets=targets)
    o = conv(o, 32 * k, stride=2, nb_classes=nb_classes, targets=targets)
    o = conv(o, 64 * k, stride=2, nb_classes=nb_classes, targets=targets)
    o = residual_block(o, 64 * k, nb_classes=nb_classes, targets=targets)
    o = residual_block(o, 64 * k, nb_classes=nb_classes, targets=targets)
    o = residual_block(o, 64 * k, nb_classes=nb_classes, targets=targets)
    o = residual_block(o, 64 * k, nb_classes=nb_classes, targets=targets)
    o = residual_block(o, 64 * k, nb_classes=nb_classes, targets=targets)
    o = upsampling(o,  32 * k, nb_classes=nb_classes, targets=targets)
    o = upsampling(o, 16 * k, nb_classes=nb_classes, targets=targets)
    o = conv(o, 3, kernel_size=9, relu=False, nb_classes=nb_classes, targets=targets)
    o = Activation('tanh')(o)
    o = Lambda(lambda x: 150*x, name='scaling')(o)
    pastiche_net = Model(input=x, output=o)
    return pastiche_net


def unet_model(img_size):
    inputs = Input(shape=(img_size,img_size,3))
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    # model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    #model.summary()


    return model


