"""
train registration network with voxelmorph based on CC
"""

# python imports
import os
import glob
import sys
import random
from argparse import ArgumentParser

# third-party imports
import tensorflow as tf
import keras.backend as keras
import numpy as np
import scipy.io as sio
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.models import load_model, Model

# project imports
import networks
import losses


vol_size = (160, 224, 192)
noftrain = 2
data_dir = '../training_data/'

training_set_filepaths = [data_dir+'l%02d.npy'%(i) for i in range(1,1+noftrain)]

train_pairs = []
for i in range(len(training_set_filepaths)):
    for j in range(len(training_set_filepaths)):
        train_pairs.append((training_set_filepaths[i],training_set_filepaths[j]))

random.shuffle(train_pairs)



def train(model,save_name, gpu_id, lr, n_iterations, reg_param, model_save_iter):

    model_dir = '../models/' + save_name
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    gpu = '/gpu:' + str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))


    # UNET filters
    nf_enc = [16,32,32,32]
    if(model == 'vm1'):
        nf_dec = [32,32,32,32,8,8,3]
    else:
        nf_dec = [32,32,32,32,32,16,16,3]

    with tf.device(gpu):
        model = networks.unet(vol_size, nf_enc, nf_dec)
        model.compile(optimizer=Adam(lr=lr), loss=[losses.cc3D(), losses.gradientLoss('l2')], loss_weights=[1.0, reg_param])
        # model.load_weights('../models/udrnet2/udrnet1_1/120000.h5')

    zeroflow = np.zeros((1, vol_size[0], vol_size[1], vol_size[2], 3))

    
    for step in range(0, n_iterations):

        sub = np.load(train_pairs[step % (noftrain ** 2)][0])
        sub = np.reshape(sub, (1,) + sub.shape + (1,))
        tmp = np.load(train_pairs[step % (noftrain ** 2)][1])
        tmp = np.reshape(tmp, (1,) + tmp.shape + (1,))
        
        train_loss = model.train_on_batch([sub, tmp], [tmp, zeroflow])

        printLoss(step, train_loss, keras.get_value(model.optimizer.lr))

        if(step % model_save_iter == 0):
            model.save(model_dir + '/' + str(step) + '.h5')
        if(step % (2*(noftrain ** 2)) == 0 and step > 0):           
            keras.set_value(model.optimizer.lr, keras.get_value(model.optimizer.lr) / 2)


def printLoss(step, train_loss, lr):
    s = 'Epoch ' + str(step // (noftrain ** 2)) + ', Total iterations ' + str(step) + ', lr= ' + str(lr) + ':'
    print(s)
    
    for i in range(len(train_loss)):
        s = '    loss %d: '%(i) + str(train_loss[i])
        print(s)
    sys.stdout.flush()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--model", type=str,dest="model", 
                        choices=['vm1','vm2'],default='vm1',
                        help="Voxelmorph-1 or 2")
    parser.add_argument("--save_name", type=str,required=True,
                        dest="save_name", help="Name of model when saving")
    parser.add_argument("--gpu", type=int,default=0,
                        dest="gpu_id", help="gpu id number")
    parser.add_argument("--lr", type=float, 
                        dest="lr", default=1e-4,help="learning rate") 
    parser.add_argument("--iters", type=int, 
                        dest="n_iterations", default=150000,
                        help="number of iterations")
    parser.add_argument("--lambda", type=float, 
                        dest="reg_param", default=1.0,
                        help="regularization parameter")
    parser.add_argument("--checkpoint_iter", type=int,
                        dest="model_save_iter", default=5000, 
                        help="frequency of model saves")

    args = parser.parse_args()
    train(**vars(args))
