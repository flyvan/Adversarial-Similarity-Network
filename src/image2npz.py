import SimpleITK as sitk
import os
import numpy as np


Path_Image='/Path to your files/ori/'
Path_Output='/Path to your files/norm/'
Path_npy='/Path to your files/npy/'



def main():
    ##---------------Set template image for affine reistration-----------------------##
    File_TemplateImg='na01_cbq.img'
    FilePath_TemplateImg=os.path.join(Path_Image,File_TemplateImg)    
    img_Template=sitk.ReadImage(FilePath_TemplateImg, sitk.sitkFloat32)    
    
    for id in range(1, 41):
        ##---------------Load subject images-----------------------##
        File_SubjectImg='na%02d_cbq.img'%(id)
        FilePath_SubjectImg=os.path.join(Path_Image,File_SubjectImg)
        
        img_Subject=sitk.ReadImage(FilePath_SubjectImg, sitk.sitkFloat32)

        
        ##--------------------------- Histogram matching ------------------------##
        matcher = sitk.HistogramMatchingImageFilter()
        matcher.SetNumberOfHistogramLevels(1024)
        matcher.SetNumberOfMatchPoints(7)
        matcher.ThresholdAtMeanIntensityOn()
        img_Subject = matcher.Execute(img_Subject,img_Template)

        ##--------------------------- Histogram matching ------------------------##

        

        File_SegImg='na%02d_seg.img'%(id)
        FilePath_SegImg=os.path.join(Path_Image,File_SegImg)
        
        img_Seg=sitk.ReadImage(FilePath_SegImg, sitk.sitkInt8)

        
        ##--------------------------- Affine Registration ---------------------##
        '''
        initialTx = sitk.CenteredTransformInitializer(img_Template, img_Subject, sitk.AffineTransform(img_Template.GetDimension()))
    
        R = sitk.ImageRegistrationMethod()
    
        R.SetShrinkFactorsPerLevel([3,2,1])
        R.SetSmoothingSigmasPerLevel([2,1,1])
        
        R.SetMetricAsMeanSquares()
        #R.SetOptimizerAsRegularStepGradientDescent(4.0, .01, 200 )
        
        R.SetOptimizerAsGradientDescent(learningRate=1.0,numberOfIterations=100,estimateLearningRate = R.EachIteration)
        R.SetOptimizerScalesFromPhysicalShift()
    
        R.SetInitialTransform(initialTx,inPlace=True)
        
        R.SetInterpolator(sitk.sitkLinear)
        
        
        outTx = R.Execute(img_Template, img_Subject)
        
        print("-------")
        print(outTx)
        print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
        print(" Iteration: {0}".format(R.GetOptimizerIteration()))
        print(" Metric value: {0}".format(R.GetMetricValue()))
        '''
        ##--------------------------- Affine Registration ---------------------##
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(img_Template)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        ##--------------------------- Size Normalization ---------------------##
        resampler.SetSize([192,224,160])
        #resampler.SetTransform(outTx)
    
        File_outputImg='l%02d.mha'%(id)
        FilePath_outputImg=os.path.join(Path_Output,File_outputImg)
            
        
        img_outputImg = resampler.Execute(img_Subject)
        caster = sitk.CastImageFilter()
        caster.SetOutputPixelType( sitk.sitkUInt8 )
        img_outputImg_int = caster.Execute( img_outputImg )
        sitk.WriteImage(img_outputImg_int, FilePath_outputImg)
        
        mat_Origin=sitk.GetArrayFromImage(img_Subject)
        print 'Original Image size:', mat_Origin.shape
        
        mat_Subject=sitk.GetArrayFromImage(img_outputImg)
        print 'Image size:', mat_Subject.shape
        #mat_Subject = mat_Subject/(maxV-minV)
        
        File_npy='l%02d.npy'%(id)
        FilePath_npy=os.path.join(Path_npy, File_npy)
        np.save(FilePath_npy, mat_Subject)
        
        
        ##-------------------------- For label images ------------------------------##
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        
        File_outputsegImg='l%02d_seg.mha'%(id)
        FilePath_outputsegImg=os.path.join(Path_Output,File_outputsegImg)
            
        
        img_outputsegImg = resampler.Execute(img_Seg)
        
        sitk.WriteImage(img_outputsegImg, FilePath_outputsegImg)
        
        
        
        mat_Seg=sitk.GetArrayFromImage(img_outputsegImg)
        print 'Image size:', mat_Seg.shape
        #mat_Subject = mat_Subject/(maxV-minV)
        
        File_npy='l%02d_seg.npy'%(id)
        FilePath_npy=os.path.join(Path_npy, File_npy)
        np.save(FilePath_npy, mat_Seg)
        ##-------------------------- For label images ------------------------------##
       
    
if __name__ == '__main__':     
    main()