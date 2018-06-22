# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 11:32:53 2018

@author: kunl
"""

from utils.transform import random_transform_generator
import cv2
import numpy as np
from utils.image import (
    TransformParameters,
    adjust_transform_for_image,
    apply_transform,
    preprocess_image,
    resize_image,
)


img = cv2.imread('../../dataset/raccoon/raccoon-1.jpg')
img = img/255.0
if img is None:
    print("can't open image")
    exit()
print(img.shape)
img2 = np.concatenate([img, np.expand_dims(np.mean(img, axis=-1),axis=-1)], axis=-1)
transform_generator = random_transform_generator(
            min_rotation=-0.1,
            max_rotation=0.1,
            min_translation=(-0.1, -0.1),
            max_translation=(0.1, 0.1),
            min_shear=-0.1,
            max_shear=0.1,
            min_scaling=(0.9, 0.9),
            max_scaling=(1.1, 1.1),
            flip_x_chance=0.5,
            flip_y_chance=0.5,
        )
image = img2
transform_parameters = TransformParameters()
transform = adjust_transform_for_image(next(transform_generator), image, transform_parameters.relative_translation)
img_ = apply_transform(transform, img,transform_parameters)
image_     = apply_transform(transform, image,transform_parameters)

cv2.imshow('img original', img[:,:,:3])
cv2.waitKey(0)
print(image_.shape)
print(image_[:,:,0])
print(img_[:,:,0])
print(image_.dtype, img.dtype)
cv2.imshow('img2', image_[:,:,0:3])
cv2.waitKey(0)
cv2.imshow('img3', img_)
cv2.waitKey(0)

cv2.destroyAllWindows()