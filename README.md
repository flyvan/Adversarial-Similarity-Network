# Reg_ASN

Unsupervised Learning for Image Registration based on Adversarial Similarity Network

[1] Fan, J., Cao, X., Xue, Z., Yap, P.T., Shen, D., Adversarial similarity network for evaluating image alignment in deep learning based registration, International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 739–746.
[2] Fan, J., Cao, X., Xue Z., Yap, P.T., Wang, Q., Shen D.,  Adversarial Learning for Mono- or Multi-Modal Registration, Medical Image Analysis, 2019; 58:101545.

This is a full-image-based implementation (Python 2.7) of the proposed method, modified from "voxelmorph" framework.
[3] Balakrishnan,  G.,  Zhao,  A.,  Sabuncu,  M.R.,  Guttag,  J.,  Dalca,  A.V., An unsupervised learning model for deformable medical image registration, Proceedings of the IEEE Conference on Computer Vision and PatternRecognition, pp. 9252–9260.

## Files
src/dense_3D_spatial_transformer.py  -  Transformational layer used in DL modal.
src/image2npz.py                     -  Generate training or testing data from images.
src/losses.py			     -  Loss functions of the networks.
src/networks.py			     -  The architecture of the networks.
src/test.py			     -  The script of testing the model.
src/train_with_CC.py		     -  Train the DL model by CC similarity metric.
src/train_with_ASN.py	      	     -  Train the DL model by adversarial similarity network.
src/train_with_SASN.py		     -  Train the DL model by adversarial similarity network with symmetric guidance.
models/				     -  The file folder of trained models. 
training_data/			     -  The file folder of training data.
testing_data/			     -  The file folder of testing data.
results/		   	     -  The file folder of testing results.


## Instructions

### Training:

1. Change dir in src/
2. python train_with_SASN.py [model_name] [gpu-id] 
examples: python train_with_SASN.py reg_SASN 0

### Testing:
1. Change dir in src/
2. python test.py [model_name] [file_name] [gpu-id] 
examples: python test.py reg_SASN demo 0
or python test.py reg_SASN 50000 0

