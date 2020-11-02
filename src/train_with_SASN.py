"""
ASNet
"""

# python imports
import os
import sys
import random
from argparse import ArgumentParser

# third-party imports
import tensorflow as tf
import keras.backend as keras
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input

# project imports
from dense_3D_spatial_transformer import Dense3DSpatialTransformer
import networks
import losses

vol_size = (160, 224, 192)
startoftrain = 1
n_train = 30


#---------------Set path to images-------------------------------#
data_dir = '../training_data/'
training_files = [data_dir+'l%02d.npy'%(i) for i in range(startoftrain, startoftrain+n_train)]

#---------------Set training pairs-------------------------------#
train_pairs = []
for i in range(n_train):
    for j in range(n_train):
        if i <> j:
            train_pairs.append((training_files[i],training_files[j]))

#---------------Set Reference pairs-----------------------------#
ref_pairs = []
for i in range(n_train):
    for j in range(n_train):
        if i <> j:
            ref_pairs.append((training_files[i],training_files[j]))

random.shuffle(train_pairs)
random.shuffle(ref_pairs)
n_pairs = len (train_pairs)
n_ref = len (ref_pairs)


def train(model,save_name, gpu_id, lr, n_iterations, reg_param, model_save_iter, alpha):

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
        
    nf = [0, 8, 16, 32, 64]

    with tf.device(gpu):
        deformer = networks.unet(vol_size, nf_enc, nf_dec)
        #deformer.compile(optimizer=Adam(lr=lr), loss=[losses.cc3D(), losses.gradientLoss('l2')], loss_weights=[1.0, reg_param])
        # model.load_weights('../models/udrnet2/udrnet1_1/120000.h5')
        
        discriminator = networks.similarity_net(vol_size, nf)
        discriminator.compile(optimizer=Adam(lr), loss='binary_crossentropy')
        discriminator.trainable = False
        
        # Build GAN model
        IA = Input(shape=vol_size + (1,))
        IB = Input(shape=vol_size + (1,))
        [WIA, DF_A2B] = deformer([IA,IB])
        [WIB, DF_B2A] = deformer([IB,IA])
        #pdb.set_trace()
        PA = discriminator([WIA, IB])
        PB = discriminator([WIB, IA])

        WWIA = Dense3DSpatialTransformer()([WIA, DF_B2A])
        WWIB = Dense3DSpatialTransformer()([WIB, DF_A2B])
                                     
        GAN = Model([IA, IB], [PA, PB, WWIA, WWIB, DF_A2B, DF_B2A])
        GAN.compile(optimizer=Adam(lr), loss=['binary_crossentropy','binary_crossentropy', 'mean_squared_error', 'mean_squared_error', 
                    losses.gradientLoss('l2'), losses.gradientLoss('l2')], loss_weights=[1.0, 1.0, 2.0, 2.0, reg_param, reg_param])
        GAN.summary()
        
    sz = PA.shape.as_list()
    sz[0] = 1
    p_one = np.ones(sz)
    p_zero = np.zeros(sz)
    zeroflow = np.zeros((1, vol_size[0], vol_size[1], vol_size[2], 3))

    for step in range(0, n_iterations):
        print 'Epoch ' + str(step // (n_pairs)) + ', Total iterations ' + str(step) + ', lr= ' + str(keras.get_value(discriminator.optimizer.lr)) + ':'
        
        sub = np.load(train_pairs[step % (n_pairs)][0])
        sub = np.reshape(sub, (1,) + sub.shape + (1,))
        tmp = np.load(train_pairs[step % (n_pairs)][1])
        tmp = np.reshape(tmp, (1,) + tmp.shape + (1,))
        
        ref_sub = np.load(ref_pairs[step % (n_ref)][0])
        ref_sub = np.reshape(ref_sub, (1,) + ref_sub.shape + (1,))
        ref_tmp = np.load(ref_pairs[step % (n_ref)][1])
        ref_tmp = np.reshape(ref_tmp, (1,) + ref_tmp.shape + (1,))
        
        
        ## ----------------  Train deformer --------------------------------##
        keras.set_value(GAN.optimizer.lr, lr)
        g_loss = GAN.train_on_batch([sub, tmp],[p_one, p_one, sub, tmp, zeroflow, zeroflow]) 
        print '  Train deformer: ' + str(g_loss[1]) + ',   ' + str(g_loss[2])
        print '  Symmetric loss: ' + str(g_loss[3]) + ',   ' + str(g_loss[4])        
        print '  Regularization: ' + str(g_loss[5]) + ',   ' + str(g_loss[6])   
        
        
        ## ----------------  Train discriminator for reference --------------##
        fused = alpha * sub + (1-alpha) * tmp
        d_loss1 = discriminator.test_on_batch([fused, tmp], p_one)
        if d_loss1 > 0.6:
            keras.set_value(discriminator.optimizer.lr, lr * 1)
        elif d_loss1 > 0.4:
            keras.set_value(discriminator.optimizer.lr, lr * 0.1)
        elif d_loss1 > 0.2:
            keras.set_value(discriminator.optimizer.lr, lr * 0.01)
        else:
            keras.set_value(discriminator.optimizer.lr, lr * 0)
        d_loss1 = discriminator.train_on_batch([fused, tmp], p_one)
        print '  Test discriminator for positive sample: ' + str(d_loss1)    
            
            
        ## ----------------  Train discriminator for deformer --------------##
        [warped, deform] = deformer.predict([sub, tmp])    
        d_loss0 = discriminator.test_on_batch([warped, tmp], p_zero)
        if d_loss0 > 0.6:
            keras.set_value(discriminator.optimizer.lr, lr * 1)
        elif d_loss0 > 0.4:
            keras.set_value(discriminator.optimizer.lr, lr * 0.1)
        elif d_loss0 > 0.2:
            keras.set_value(discriminator.optimizer.lr, lr * 0.01)
        else:
            keras.set_value(discriminator.optimizer.lr, lr * 0)
        d_loss0 = discriminator.train_on_batch([warped, tmp], p_zero)
        print '  Test discriminator for negative sample: ' + str(d_loss0)                

            
        if(step % model_save_iter == 0):
            deformer.save(model_dir + '/' + str(step) + '.h5')
            
        #if(step % (20 * n_pairs) == 0 and step > 0):           
            #lr = lr / 2
            #alpha = np.abs(alpha - 0.05)

        sys.stdout.flush()
    


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--model", type=str,dest="model", 
                        choices=['vm1','vm2'],default='vm1',
                        help="Voxelmorph-1 or 2")
    
    parser.add_argument("--save_name", type=str,required=True,
                        dest="save_name", 
                        help="Name of model when saving")
    
    parser.add_argument("--gpu", type=int,default=0,
                        dest="gpu_id", help="gpu id number")
    
    parser.add_argument("--lr", type=float, 
                        dest="lr", default=1e-4,
                        help="learning rate") 
    
    parser.add_argument("--iters", type=int, 
                        dest="n_iterations", default=150000,
                        help="number of iterations")
    
    parser.add_argument("--lambda", type=float, 
                        dest="reg_param", default=2e2,
                        help="regularization parameter")
    #1e3 for gradientLoss      1e6 for Jaccobian loss
    
    parser.add_argument("--checkpoint_iter", type=int,
                        dest="model_save_iter", default=5000, 
                        help="frequency of model saves")
    
    
    parser.add_argument("--alpha", type=float,
                        dest="alpha", default=0.2, 
                        help="degree of image fusion of reference images")

    args = parser.parse_args()
    train(**vars(args))
