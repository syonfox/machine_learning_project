from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Lambda, UpSampling2D
from keras.layers.merge import concatenate
from keras.models import Model,Sequential
from layers import InputNormalize,VGGNormalize,ReflectionPadding2D,Denormalize,conv_bn_relu,res_conv,dconv_bn_nolinear, UnPooling2D
from keras_contrib.layers.normalization import InstanceNormalization
from loss import StyleReconstructionRegularizer,FeatureReconstructionRegularizer,TVRegularizer
from keras import backend as K
from VGG16 import VGG16
import img_util


from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation

def image_transform_net(img_width,img_height,tv_weight=1):
    x = Input(shape=(img_width,img_height,3))
    a = InputNormalize()(x)
    a = ReflectionPadding2D(padding=(40,40),input_shape=(img_width,img_height,3))(a)
    a = conv_bn_relu(32, 9, 9, stride=(1,1))(a)
    a = conv_bn_relu(64, 9, 9, stride=(2,2))(a)
    a = conv_bn_relu(128, 3, 3, stride=(2,2))(a)
    for i in range(5):
        a = res_conv(128,3,3)(a)
    a = dconv_bn_nolinear(64,3,3)(a)
    a = dconv_bn_nolinear(32,3,3)(a)
    a = dconv_bn_nolinear(3,9,9,stride=(1,1),activation="tanh")(a)
    # Scale output to range [0, 255] via custom Denormalize layer
    y = Denormalize(name='transform_output')(a)
    
    model = Model(inputs=x, outputs=y)
    
    if tv_weight > 0:
        add_total_variation_loss(model.layers[-1],tv_weight)
        
    return model 
def conv_n(nb_filter, kernal_size, activation = 'relu', padding = "same",kernel_initializer = 'he_normal'):
    def conv_func(x):


        if(padding=="reflect"):
            # x = UnPooling2D(size=(1,1))(x)
            x = ReflectionPadding2D(padding=(1,1))(x)
            x = Conv2D(nb_filter, kernal_size, strides=(1,1),padding="valid", kernel_initializer=kernel_initializer)(x)

        else:
            x = Conv2D(nb_filter, kernal_size, strides=(1,1),padding=padding, kernel_initializer=kernel_initializer)(x)

        x = InstanceNormalization()(x)
        #x = BatchNormalization()(x)
        #x = LeakyReLU(0.2)(x)
        x = Activation(activation)(x)
        return x
    return conv_func

def unet(img_width,img_height,tv_weight=1):

    inputs = Input(shape=(img_width,img_height,3))
    conv1 = conv_n(36, 3, activation = 'relu', padding = 'reflect', kernel_initializer = 'he_normal')(inputs)
    conv1 = conv_n(36, 3, activation = 'relu', padding = 'reflect', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_n(64, 3, activation = 'relu', padding = 'reflect', kernel_initializer = 'he_normal')(pool1)
    conv2 = conv_n(64, 3, activation = 'relu', padding = 'reflect', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_n(128, 3, activation = 'relu', padding = 'reflect', kernel_initializer = 'he_normal')(pool2)
    conv3 = conv_n(128, 3, activation = 'relu', padding = 'reflect', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_n(256, 3, activation = 'relu', padding = 'reflect', kernel_initializer = 'he_normal')(pool3)
    conv4 = conv_n(256, 3, activation = 'relu', padding = 'reflect', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = conv_n(512, 3, activation = 'relu', padding = 'reflect', kernel_initializer = 'he_normal')(pool4)
    conv5 = conv_n(512, 3, activation = 'relu', padding = 'reflect', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = conv_n(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = conv_n(256, 3, activation = 'relu', padding = 'reflect', kernel_initializer = 'he_normal')(merge6)
    conv6 = conv_n(256, 3, activation = 'relu', padding = 'reflect', kernel_initializer = 'he_normal')(conv6)

    up7 = conv_n(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = conv_n(128, 3, activation = 'relu', padding = 'reflect', kernel_initializer = 'he_normal')(merge7)
    conv7 = conv_n(128, 3, activation = 'relu', padding = 'reflect', kernel_initializer = 'he_normal')(conv7)

    up8 = conv_n(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = conv_n(64, 3, activation = 'relu', padding = 'reflect', kernel_initializer = 'he_normal')(merge8)
    conv8 = conv_n(64, 3, activation = 'relu', padding = 'reflect', kernel_initializer = 'he_normal')(conv8)

    up9 = conv_n(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = conv_n(32, 3, activation = 'relu', padding = 'reflect', kernel_initializer = 'he_normal')(merge9)
    conv9 = conv_n(32, 3, activation = 'relu', padding = 'reflect', kernel_initializer = 'he_normal')(conv9)
    conv9 = conv_n(9, 3, activation = 'relu', padding = 'reflect', kernel_initializer = 'he_normal')(conv9)

    conv10 = conv_n(3, 1, activation = 'sigmoid')(conv9)

    o = Lambda(lambda x: 150*x, name='scaling')(conv10)

    model = Model(input = inputs, output = o)

    # model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    #model.summary()

    if tv_weight > 0:
        add_total_variation_loss(model.layers[-1],tv_weight)

    return model


# def unet(img_width,img_height,tv_weight=1):
#
#     inputs = Input(shape=(img_width,img_height,3))
#     conv1 = Conv2D(36, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
#     conv1 = Conv2D(36, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#     conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
#     conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#     conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
#     conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#     conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
#     conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
#     drop4 = Dropout(0.5)(conv4)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
#
#     conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
#     conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
#     drop5 = Dropout(0.5)(conv5)
#
#     up6 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
#     merge6 = concatenate([drop4,up6], axis = 3)
#     conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
#     conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
#
#     up7 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
#     merge7 = concatenate([conv3,up7], axis = 3)
#     conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
#     conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
#
#     up8 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
#     merge8 = concatenate([conv2,up8], axis = 3)
#     conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
#     conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
#
#     up9 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
#     merge9 = concatenate([conv1,up9], axis = 3)
#     conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
#     conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
#     conv9 = Conv2D(9, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
#
#     conv10 = Conv2D(3, 1, activation = 'sigmoid')(conv9)
#
#     o = Lambda(lambda x: 150*x, name='scaling')(conv10)
#
#     model = Model(input = inputs, output = o)
#
#     # model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
#
#     #model.summary()
#
#     if tv_weight > 0:
#         add_total_variation_loss(model.layers[-1],tv_weight)
#
#     return model
#

def loss_net(x_in, trux_x_in,width, height,style_image_path,content_weight,style_weight):
    # Append the initial input to the FastNet input to the VGG inputs
    x = concatenate([x_in, trux_x_in], axis=0)
    
    # Normalize the inputs via custom VGG Normalization layer
    x = VGGNormalize(name="vgg_normalize")(x)

    vgg = VGG16(include_top=False,input_tensor=x)

    vgg_output_dict = dict([(layer.name, layer.output) for layer in vgg.layers[-18:]])
    vgg_layers = dict([(layer.name, layer) for layer in vgg.layers[-18:]])

    if style_weight > 0:
        add_style_loss(vgg,style_image_path , vgg_layers, vgg_output_dict, width, height,style_weight)   

    if content_weight > 0:
        add_content_loss(vgg_layers,vgg_output_dict,content_weight)

    # Freeze all VGG layers
    for layer in vgg.layers[-19:]:
        layer.trainable = False

    return vgg

def add_style_loss(vgg,style_image_path,vgg_layers,vgg_output_dict,img_width, img_height,weight):
    style_img = img_util.preprocess_image(style_image_path, img_width, img_height)
    print('Getting style features from VGG network.')

    style_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3']

    style_layer_outputs = []

    for layer in style_layers:
        style_layer_outputs.append(vgg_output_dict[layer])

    vgg_style_func = K.function([vgg.layers[-19].input], style_layer_outputs)

    style_features = vgg_style_func([style_img])

    # Style Reconstruction Loss
    for i, layer_name in enumerate(style_layers):
        layer = vgg_layers[layer_name]

        feature_var = K.variable(value=style_features[i][0])
        style_loss = StyleReconstructionRegularizer(
                            style_feature_target=feature_var,
                            weight=weight)(layer)

        layer.add_loss(style_loss)

def add_content_loss(vgg_layers,vgg_output_dict,weight):
    # Feature Reconstruction Loss
    content_layer = 'block3_conv3'
    content_layer_output = vgg_output_dict[content_layer]

    layer = vgg_layers[content_layer]
    content_regularizer = FeatureReconstructionRegularizer(weight)(layer)
    layer.add_loss(content_regularizer)


def add_total_variation_loss(transform_output_layer,weight):
    # Total Variation Regularization
    layer = transform_output_layer  # Output layer
    tv_regularizer = TVRegularizer(weight)(layer)
    layer.add_loss(tv_regularizer)

