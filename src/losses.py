
# Third party inports
import tensorflow as tf
import numpy as np

# batch_sizexheightxwidthxdepthxchan


def diceLoss(y_true, y_pred):
    top = 2*tf.reduce_sum(y_true * y_pred, [1, 2, 3])
    bottom = tf.maximum(tf.reduce_sum(y_true+y_pred, [1, 2, 3]), 1e-5)
    dice = tf.reduce_mean(top/bottom)
    return -dice

def topology_preservation_loss(min=0.1, max=10.0):
    def regularize(x):
        return (-x + min) * tf.cast(x<=min, tf.float32) + (x - max) * tf.cast(x>=max, tf.float32)
    def loss(y_true, y_pred):
        filter_np = np.array([[[-1,1]]],dtype=np.float32)
        filter_np_x = np.swapaxes(filter_np,0,2)
        filter_np_y = np.swapaxes(filter_np,1,2)
        filter_np = np.expand_dims(filter_np,-1)
        filter_np = np.expand_dims(filter_np,-1)
        filter_np_x = np.expand_dims(filter_np_x,-1)
        filter_np_x = np.expand_dims(filter_np_x,-1)
        filter_np_y = np.expand_dims(filter_np_y,-1)
        filter_np_y = np.expand_dims(filter_np_y,-1)
        filter_x = tf.constant(filter_np_x)
        filter_y = tf.constant(filter_np_y)
        filter_z = tf.constant(filter_np)
        f = tf.expand_dims(y_pred[:,:,:,:,0],4)
        g = tf.expand_dims(y_pred[:,:,:,:,1],4)
        e = tf.expand_dims(y_pred[:,:,:,:,2],4)
        # foward gradient
        fxf = tf.nn.conv3d(tf.pad(f,[[0,0],[0,1],[0,0],[0,0],[0,0]]), filter_x, strides=[1,1,1,1,1], padding='VALID') + 1.0
        fyf = tf.nn.conv3d(tf.pad(f,[[0,0],[0,0],[0,1],[0,0],[0,0]]), filter_y, strides=[1,1,1,1,1], padding='VALID')
        fzf = tf.nn.conv3d(tf.pad(f,[[0,0],[0,0],[0,0],[0,1],[0,0]]), filter_z, strides=[1,1,1,1,1], padding='VALID')
        gxf = tf.nn.conv3d(tf.pad(g,[[0,0],[0,1],[0,0],[0,0],[0,0]]), filter_x, strides=[1,1,1,1,1], padding='VALID')
        gyf = tf.nn.conv3d(tf.pad(g,[[0,0],[0,0],[0,1],[0,0],[0,0]]), filter_y, strides=[1,1,1,1,1], padding='VALID') + 1.0
        gzf = tf.nn.conv3d(tf.pad(g,[[0,0],[0,0],[0,0],[0,1],[0,0]]), filter_z, strides=[1,1,1,1,1], padding='VALID')
        exf = tf.nn.conv3d(tf.pad(e,[[0,0],[0,1],[0,0],[0,0],[0,0]]), filter_x, strides=[1,1,1,1,1], padding='VALID')
        eyf = tf.nn.conv3d(tf.pad(e,[[0,0],[0,0],[0,1],[0,0],[0,0]]), filter_y, strides=[1,1,1,1,1], padding='VALID')
        ezf = tf.nn.conv3d(tf.pad(e,[[0,0],[0,0],[0,0],[0,1],[0,0]]), filter_z, strides=[1,1,1,1,1], padding='VALID') + 1.0

        # backward gradient
        fxb = tf.nn.conv3d(tf.pad(f,[[0,0],[1,0],[0,0],[0,0],[0,0]]), filter_x, strides=[1,1,1,1,1], padding='VALID') + 1.0
        fyb = tf.nn.conv3d(tf.pad(f,[[0,0],[0,0],[1,0],[0,0],[0,0]]), filter_y, strides=[1,1,1,1,1], padding='VALID')
        fzb = tf.nn.conv3d(tf.pad(f,[[0,0],[0,0],[0,0],[1,0],[0,0]]), filter_z, strides=[1,1,1,1,1], padding='VALID')
        gxb = tf.nn.conv3d(tf.pad(g,[[0,0],[1,0],[0,0],[0,0],[0,0]]), filter_x, strides=[1,1,1,1,1], padding='VALID')
        gyb = tf.nn.conv3d(tf.pad(g,[[0,0],[0,0],[1,0],[0,0],[0,0]]), filter_y, strides=[1,1,1,1,1], padding='VALID') + 1.0
        gzb = tf.nn.conv3d(tf.pad(g,[[0,0],[0,0],[0,0],[1,0],[0,0]]), filter_z, strides=[1,1,1,1,1], padding='VALID')
        exb = tf.nn.conv3d(tf.pad(e,[[0,0],[1,0],[0,0],[0,0],[0,0]]), filter_x, strides=[1,1,1,1,1], padding='VALID')
        eyb = tf.nn.conv3d(tf.pad(e,[[0,0],[0,0],[1,0],[0,0],[0,0]]), filter_y, strides=[1,1,1,1,1], padding='VALID')
        ezb = tf.nn.conv3d(tf.pad(e,[[0,0],[0,0],[0,0],[1,0],[0,0]]), filter_z, strides=[1,1,1,1,1], padding='VALID') + 1.0
        
        J_fff = regularize(fxf*gyf*ezf + fyf*gzf*exf + fzf*gxf*eyf - fzf*gyf*exf - fyf*gxf*ezf - fxf*gzf*eyf)
        J_bff = regularize(fxb*gyf*ezf + fyf*gzf*exb + fzf*gxb*eyf - fzf*gyf*exb - fyf*gxb*ezf - fxb*gzf*eyf)
        J_fbf = regularize(fxf*gyb*ezf + fyb*gzf*exf + fzf*gxf*eyb - fzf*gyb*exf - fyb*gxf*ezf - fxf*gzf*eyb)
        J_ffb = regularize(fxf*gyf*ezb + fyf*gzb*exf + fzb*gxf*eyf - fzb*gyf*exf - fyf*gxf*ezb - fxf*gzb*eyf)
        J_bbb = regularize(fxb*gyb*ezb + fyb*gzb*exb + fzb*gxb*eyb - fzb*gyb*exb - fyb*gxb*ezb - fxb*gzb*eyb)
        J_fbb = regularize(fxf*gyb*ezb + fyb*gzb*exf + fzb*gxf*eyb - fzb*gyb*exf - fyb*gxf*ezb - fxf*gzb*eyb)
        J_bfb = regularize(fxb*gyf*ezb + fyf*gzb*exb + fzb*gxb*eyf - fzb*gyf*exb - fyf*gxb*ezb - fxb*gzb*eyf)
        J_bbf = regularize(fxb*gyb*ezf + fyb*gzf*exb + fzf*gxb*eyb - fzf*gyb*exb - fyb*gxb*ezf - fxb*gzf*eyb)


        corner_jacob = tf.reduce_mean(J_fff)+tf.reduce_mean(J_bff)+tf.reduce_mean(J_fbf)+tf.reduce_mean(J_ffb)+tf.reduce_mean(J_bbb)+tf.reduce_mean(J_fbb)+tf.reduce_mean(J_bfb)+tf.reduce_mean(J_bbf)
        
        #return tf.reduce_mean(J_fff+J_bff+J_fbf+J_ffb+J_bbb+J_fbb+J_bfb+J_bbf) / 8.0
        return corner_jacob / 8
    return loss

def gradientLoss(penalty='l1'):
    def loss(y_true, y_pred):
        dy = y_pred[:, 1:, :, :, :] - y_pred[:, :-1, :, :, :]
        dx = y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :]
        dz = y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :]
        
        m = tf.cast(tf.cast((dy <= -1), tf.int32) * 10 + tf.cast((dy > -1), tf.int32), tf.float32)
        dy = dy * m
        m = tf.cast(tf.cast((dx <= -1), tf.int32) * 10 + tf.cast((dx > -1), tf.int32), tf.float32)
        dx = dx * m
        m = tf.cast(tf.cast((dz <= -1), tf.int32) * 10 + tf.cast((dz > -1), tf.int32), tf.float32)
        dz = dz * m
        
        dy = tf.abs(dy)
        dx = tf.abs(dx)
        dz = tf.abs(dz)

        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz
        d = tf.reduce_mean(dx)+tf.reduce_mean(dy)+tf.reduce_mean(dz)
        
        return d/3.0
   
    return loss


def gradientLoss2D():
    def loss(y_true, y_pred):
        dy = tf.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :])
        dx = tf.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])

        dy = dy * dy
        dx = dx * dx

        d = tf.reduce_mean(dx)+tf.reduce_mean(dy)
        return d/2.0

    return loss


def cc3D(win=[9, 9, 9], voxel_weights=None):
    def loss(I, J):
        I2 = I*I
        J2 = J*J
        IJ = I*J

        filt = tf.ones([win[0], win[1], win[2], 1, 1])

        I_sum = tf.nn.conv3d(I, filt, [1, 1, 1, 1, 1], "SAME")
        J_sum = tf.nn.conv3d(J, filt, [1, 1, 1, 1, 1], "SAME")
        I2_sum = tf.nn.conv3d(I2, filt, [1, 1, 1, 1, 1], "SAME")
        J2_sum = tf.nn.conv3d(J2, filt, [1, 1, 1, 1, 1], "SAME")
        IJ_sum = tf.nn.conv3d(IJ, filt, [1, 1, 1, 1, 1], "SAME")

        win_size = win[0]*win[1]*win[2]
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var+1e-5)

        # if(voxel_weights is not None):
        #	cc = cc * voxel_weights
        
        return -1.0*tf.reduce_mean(cc)

    
    return loss


def cc2D(win=[9, 9]):
    def loss(I, J):
        I2 = tf.multiply(I, I)
        J2 = tf.multiply(J, J)
        IJ = tf.multiply(I, J)

        sum_filter = tf.ones([win[0], win[1], 1, 1])

        I_sum = tf.nn.conv2d(I, sum_filter, [1, 1, 1, 1], "SAME")
        J_sum = tf.nn.conv2d(J, sum_filter, [1, 1, 1, 1], "SAME")
        I2_sum = tf.nn.conv2d(I2, sum_filter, [1, 1, 1, 1], "SAME")
        J2_sum = tf.nn.conv2d(J2, sum_filter, [1, 1, 1, 1], "SAME")
        IJ_sum = tf.nn.conv2d(IJ, sum_filter, [1, 1, 1, 1], "SAME")

        win_size = win[0]*win[1]

        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var + np.finfo(float).eps)
        return -1.0*tf.reduce_mean(cc)
    return loss
