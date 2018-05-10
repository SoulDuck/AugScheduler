import imgaug as ia
from imgaug import augmenters as iaa
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#iaa.Fliplr
#iaa.Flipud
#iaa.Crop()
#iaa.CropAndPad
img=Image.open('./sample_fundus.png')
img=np.asarray(img)

iaa.Sometimes(0.5 , iaa.GaussianBlur(sigma=(0 ,0.5)))

iaa.AdditiveGaussianNoise(loc=0 , scale=(0.0 , 0.05*255) , per_channel=0.5),


#Crop
iaa.Crop(percent=(0,0.1))

#Blur
iaa.GaussianBlur((0 , 3.0))
iaa.AverageBlur((2, 7))
iaa.MedianBlur((3 , 11))

iaa.Sharpen(alpha=(0,1.0) , lightness=(0.75 , 1.5))
iaa.Emboss(alpha=(0,1.0) , strength=(0,2.0))
iaa.EdgeDetect(alpha=(0,0.7))
iaa.DirectedEdgeDetect(alpha=(0,0.7) , direction=(0.0 , 1.0))

#Noise
iaa.AdditiveGaussianNoise(loc=0 , scale = (0.0 , 0.05*255) , per_channel=0.5)
iaa.Dropout((0.01 , 0.1) , per_channel=0.5)
iaa.CoarseDropout((0.03 , 0.15 ) ,size_percent=(0.02 , 0.05) , per_channel=0.2)

# Color Change
iaa.Invert(0.05  ,per_channel=True)
iaa.Add((-10, 10) , per_channel=0.5)
iaa.Multiply((0.8 , 1.2) , per_channel=0.2),
iaa.ContrastNormalization((0.5,2.0) , per_channel=0.5)

#
iaa.ElasticTransformation(alpha=(0.5 , 3.5) , sigma=0.25)
iaa.Superpixels(p_replace=(0 , 1.0) , n_segments= (20,200))

#Affine Augmentation
iaa.PiecewiseAffine(scale= (0.01 , 0.05))
iaa.Affine(scale={"x" : (0.8 ,1.2) , "y" : (0.8 , 1.2)}  , translate_percent={"x" :(0.8 , 1.2 ) , "y":(0.8,1.2) },
           rotate=(-25,25) , shear=(-8 , 8))



seq=iaa.Sequential([
    #iaa.Sometimes(0.5 , iaa.GaussianBlur(sigma=(0 ,0.5)))
    #iaa.GaussianBlur(sigma=(1))
    #iaa.Multiply((0.8 , 1.5) , per_channel=0.5)
    #iaa.Affine( scale={"x" :(0.8 ,1.2) , "y":(0.8 , 1.2)}  ,translate_percent = {"x":(-0.2 , 0.2 ) ,"y":(-0.2,0.2)},
    #   rotate=(-25,25) , shear=(-8 , 8))
    #iaa.Superpixels(p_replace=(0 , 1.0) , n_segments= (20,200))
    #iaa.GaussianBlur((0 , 3.0))
    #iaa.MedianBlur((3 , 11))
    #iaa.Sharpen(alpha=(0,1.0) , lightness=(0.75 , 1.5))
    iaa.Emboss(alpha=(0,1.0) , strength=(0,2.0))
    #iaa.EdgeDetect(alpha=(0,0.7))
    #iaa.AdditiveGaussianNoise(loc=0 , scale = (0.0 , 0.05*255) , per_channel=0.5)
    #iaa.CoarseDropout((0.03 , 0.15 ) ,size_percent=(0.02 , 0.05) , per_channel=0.2)
    #iaa.Add((-10, 10) , per_channel=0.5)
    #iaa.ElasticTransformation(alpha=(1.0) , sigma=0.25)
    #iaa.PiecewiseAffine(scale= (0.01 , 0.05))
    #iaa.ContrastNormalization((0.5,2.0) , per_channel=0.5)
] , random_order=True)



augimg=seq.augment_image(img)
plt.imshow(augimg)
plt.show()


augimg=seq.augment_image(img)
plt.imshow(augimg)
plt.show()


augimg=seq.augment_image(img)
plt.imshow(augimg)
plt.show()

augimg=seq.augment_image(img)
plt.imshow(augimg)
plt.show()

