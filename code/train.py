from keras.layers import Input, merge
from keras.models import Model,Sequential
from layers import VGGNormalize,ReflectionPadding2D,Denormalize,conv_bn_relu,res_conv,dconv_bn_nolinear
from loss import dummy_loss,StyleReconstructionRegularizer,FeatureReconstructionRegularizer,TVRegularizer
from keras.optimizers import Adam, SGD,RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from scipy.misc import imsave
import time
import numpy as np
import argparse
import h5py
import  tensorflow as tf
from keras.callbacks import TensorBoard
from scipy import ndimage
import json

import nets

def config_gpu(gpu, allow_growth):
    # Choosing gpu
    if gpu == '-1':
        config = tf.ConfigProto(device_count ={'GPU': 0})
    else:
        if gpu == 'all' or gpu == '':
            gpu = ''
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = gpu
    if allow_growth == True:
        config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)

def display_img(i,x,out_path,is_val=False):
    # save current generated image
    img = x #deprocess_image(x)
    if is_val:
        #img = ndimage.median_filter(img, 3)

        fname = '%s%d_val.png' % (out_path,i)
    else:
        fname = '%s%d_x.png' % (out_path,i)
    imsave(fname, img)
    print('Image saved as', fname)


def main(args):
    style_weight= args.style_weight
    content_weight= args.content_weight
    tv_weight= args.tv_weight
    style= args.style_path
    img_width = img_height =  args.image_size
    save_interval = args.save_interval
    style_image_path = style

    if(args.model == "unet"):
        net = nets.unet(img_width,img_height,tv_weight)
    else:
        net = nets.image_transform_net(img_width,img_height,tv_weight)
    model = nets.loss_net(net.output,net.input,img_width,img_height,style_image_path,content_weight,style_weight)
    model.summary()


    num_epochs = 2
    batch_size =  1
    learning_rate = 1e-3 #1e-3
    optimizer = Adam() # Adam(lr=learning_rate,beta_1=0.99)

    model.compile(optimizer,  dummy_loss)  # Dummy loss since we are learning from regularizes

    #datagen = ImageDataGenerator()

    dummy_y = np.zeros((batch_size,img_width,img_height,3)) # Dummy output, not used since we use regularizers to train

 

    #model.load_weights(style+'_weights.h5',by_name=False)

    skip_to = 0
    # X = h5py.File(args.train_path, 'r')['train2014']['images']
    # dataset_size = X.shape[0]

    # Load the data
    X = h5py.File(args.train_path, 'r')['train2014']['images']
    dataset_size = X.shape[0]
    batches_per_epoch = int(np.ceil(dataset_size / batch_size))
    batch_idx = 0



    i=0
    t1 = time.time()
    #for x in datagen.flow_from_directory(args.train_path, class_mode=None, batch_size=train_batchsize,
    #    target_size=(img_width, img_height), shuffle=False):
    epoch = 0
    num_iterations = num_epochs * dataset_size
    # for it in range(args.num_iterations):

    log = open(args.output_path + "log.txt", "w+")

    #log.write("Training "+ num_epochs+ " epochs in "+ num_iterations+ "interations using model: "+ args.model)
    #log.write("<->")
    # log.write("Epochs:", num_epochs)
    # log.write("Iterations:", num_iterations)
    # log.write("StyleWeight:", style_weight)
    # log.write("ContentWeight:", content_weight)
    # log.write("TVWeight:", tv_weight)

    info = {'epochs': num_epochs, 'interations': num_iterations, 'style_weight': style_weight, 'content_weight': content_weight, 'tv_weight':tv_weight, 'learning_rate':learning_rate}
    log.write(json.dumps(info))


    for it in range(num_iterations):
        if batch_idx == batches_per_epoch:
            print('Epoch done. Going back to the beginning...')
            epoch += 1
            batch_idx = 0

        # Get the batch
        idx = batch_size * batch_idx
        x = X[idx:idx+batch_size] #batch
        #batch = preprocess_input(batch)
        batch_idx += 1
        hist = model.train_on_batch(x, dummy_y)

        if it % 100 == 0:
            #print(hist,(time.time() -t1))
            #hist.
            print("Epoch:", epoch, ", Iteration:", it,", Loss: ",hist,", Time: ",(time.time() -t1))
            itlog = { 'epoch': epoch, 'interation': it, 'loss':float(hist),'time': float((time.time())), "delta_time": float((time.time()-t1))}
            #print(itlog)
            log.write(json.dumps(itlog))
            t1 = time.time()

        if it % 500 == 0:
            print("Validating/Saving...Epoch:", epoch, ", Iteration:", it,", Loss: ",hist)
            #print()
            val_x = net.predict(x)

            display_img(it, x[0], args.output_path)
            display_img(it, val_x[0],args.output_path, True)
            model.save_weights(args.output_path+'weights.h5')
            print("Done.")
        if it % save_interval == 0:
            print("Saving...")
            model.save_weights(args.output_path+str(it)+'_weights.h5')
            print("Done.")



    # for x in datagen.flow(X, batch_size=train_batchsize, shuffle=False):
    #     if i > nb_epoch:
    #         break
    #
    #     if i < skip_to:
    #         i+=train_batchsize
    #         if i % 1000 ==0:
    #             print("skip to: %d" % i)
    #
    #         continue
    #
    #
    #     hist = model.(x, dummy_y)
    #
    #     if i % 50 == 0:
    #         print(hist,(time.time() -t1))
    #         t1 = time.time()
    #
    #     if i % 500 == 0:
    #         print("epoc: ", i)
    #         val_x = net.predict(x)
    #
    #         display_img(i, x[0], args.output_path)
    #         display_img(i, val_x[0],args.output_path, True)
    #         model.save_weights(args.output_path+'weights.h5')
    #
    #     i+=train_batchsize



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-time style transfer')
        
    parser.add_argument('--style_path', '-s', type=str, required=True,
                        help='style image file name without extension')
    parser.add_argument('--train_path', '-t', type=str, default='./data/ms-coco-256.h5',
                        help='style image file name without extension')
          
    parser.add_argument('--output_path', '-o', default="./save/", type=str,
                        help='output model file path without extension')
    parser.add_argument('--tv_weight', default=1e-6, type=float,
                        help='weight of total variation regularization according to the paper to be set between 10e-4 and 10e-6.')
    parser.add_argument('--content_weight', default=1.0, type=float)
    parser.add_argument('--style_weight', default=4.0, type=float)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument("--model", type=str, default="default")
    parser.add_argument("--save_interval", type=int, default=25000)
    #config_gpu(args.gpu, args.allow_growth)
    args = parser.parse_args()
    main(args)
