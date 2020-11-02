# py imports
import sys
import SimpleITK as sitk

# third party
import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session
from scipy.interpolate import interpn

# project
import networks


##----------------Load your dataset for testing-------------------##
startoftest = 31
noftest = 2

data_dir = '../testing_data/'
output_dir = '../results/'
subjcet_filepaths = [data_dir+'l%02d.npy'%(i) for i in range(startoftest, startoftest+noftest)]
seg_filepaths = [data_dir+'l%02d_seg.npy'%(i) for i in range(startoftest, startoftest+noftest)]


def dice(vol1, vol2, labels=None, nargout=1):
    '''
    Dice [1] volume overlap metric

    The default is to *not* return a measure for the background layer (label = 0)

    [1] Dice, Lee R. "Measures of the amount of ecologic association between species."
    Ecology 26.3 (1945): 297-302.

    Parameters
    ----------
    vol1 : nd array. The first volume (e.g. predicted volume)
    vol2 : nd array. The second volume (e.g. "true" volume)
    labels : optional vector of labels on which to compute Dice.
        If this is not provided, Dice is computed on all non-background (non-0) labels
    nargout : optional control of output arguments. if 1, output Dice measure(s).
        if 2, output tuple of (Dice, labels)

    Output
    ------
    if nargout == 1 : dice : vector of dice measures for each labels
    if nargout == 2 : (dice, labels) : where labels is a vector of the labels on which
        dice was computed
    '''
    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))
        labels = np.delete(labels, np.where(labels == 0))  # remove background

    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        top = 2 * np.sum(np.logical_and(vol1 == lab, vol2 == lab))
        bottom = np.sum(vol1 == lab) + np.sum(vol2 == lab)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom

    if nargout == 1:
        return dicem
    else:
        return (dicem, labels)



def test(model_name, iter_num, gpu_id, vol_size=(160,224,192), nf_enc=[16,32,32,32], nf_dec=[32,32,32,32,8,8,3]):
    """
    test
    nf_enc and nf_dec
    #nf_dec = [32,32,32,32,32,16,16,3]
    # This needs to be changed. Ideally, we could just call load_model, and we wont have to
    # specify the # of channels here, but the load_model is not working with the custom loss...
    """  

    gpu = '/gpu:' + str(gpu_id)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

	# load weights of model
    with tf.device(gpu):
        net = networks.unet(vol_size, nf_enc, nf_dec)
        net.load_weights('../models/' + model_name +
                         '/' + str(iter_num) + '.h5')

    xx = np.arange(vol_size[1])
    yy = np.arange(vol_size[0])
    zz = np.arange(vol_size[2])
    grid = np.rollaxis(np.array(np.meshgrid(xx, yy, zz)), 0, 4)
    
    sum_dice = 0

    for i in range(0, noftest):
        X_vol = np.load(subjcet_filepaths[i])
        X_vol = np.reshape(X_vol, (1,) + X_vol.shape + (1,))
        X_seg = np.load(seg_filepaths[i])
        #X_seg = np.reshape(X_seg, (1,) + X_seg.shape + (1,))
	
        for j in range(0, noftest):
            Y_vol = np.load(subjcet_filepaths[j])
            Y_vol = np.reshape(Y_vol, (1,) + Y_vol.shape + (1,))
            Y_seg = np.load(seg_filepaths[j])
            #Y_seg = np.reshape(Y_seg, (1,) + Y_seg.shape + (1,))
            
            with tf.device(gpu):
                pred = net.predict([X_vol, Y_vol])
                
            # Warp segments with flow
            flow = pred[1][0, :, :, :, :]
            sample = flow+grid
            sample = np.stack((sample[:, :, :, 1], sample[:, :, :, 0], sample[:, :, :, 2]), 3)
            warp_seg = interpn((yy, xx, zz), X_seg, sample, method='nearest', bounds_error=False, fill_value=0)
                
            #print 'X Image size:', X_seg.shape
            #print 'warped Image size:', warp_seg.shape
            #print 'Y Image size:', Y_seg.shape
                
            vals, lnames = dice(warp_seg, Y_seg, nargout=2)
            print '%dth image to %dth image --- '%(i+startoftest, j+startoftest),'mean: ', np.mean(vals), 'std: ', np.std(vals)
            sum_dice = sum_dice + np.mean(vals)
            
            mat_Deform = pred[1][0, :, :, :, :]
        
            img_Deform=sitk.GetImageFromArray(mat_Deform)        
            print 'img_Deform shape, ',mat_Deform.shape
            outputfilename=output_dir+'deformations/'+'Deform%02dto%02d.mha'%(i+startoftest,j+startoftest)
            
            sitk.WriteImage(img_Deform, outputfilename)       

    
            mat_warpimage = pred[0][0, :, :, :, 0]
            img_Warp=sitk.GetImageFromArray(mat_warpimage)   
            caster = sitk.CastImageFilter()
            caster.SetOutputPixelType( sitk.sitkInt16 )
            img_Warp = caster.Execute( img_Warp )        
            outputfilename=output_dir+'warp_images/'+'na%02dto%02d.mha'%(i+startoftest,j+startoftest)
            sitk.WriteImage(img_Warp, outputfilename)
            
            img_Warpseg = sitk.GetImageFromArray(warp_seg)
            img_Warpseg = caster.Execute( img_Warpseg )        
            outputfilename=output_dir+'warp_images/'+'seg%02dto%02d.mha'%(i+startoftest,j+startoftest)
            sitk.WriteImage(img_Warpseg, outputfilename)
    
    print 'Average Dice ratio: ', sum_dice/(noftest ** 2)

if __name__ == "__main__":
	test(sys.argv[1], sys.argv[2], sys.argv[3])
