import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize
from skimage.exposure import equalize_adapthist, equalize_hist
import math
from skimage import measure 

from skimage.measure import find_contours
from ellipse import LsqEllipse
from matplotlib.patches import Ellipse

from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')
from scipy.ndimage import morphology
import tensorflow

from Promise12_Final.pre_processing_data import *
from Promise12_Final.deep_learning_model import *
from Promise12_Final.exp_and_results import *
from ProstateX_Final.Model.grad_cam_code import *
from ProstateX_Final.Model.common_functions import *

import numpy as np
from myshow import *

from ellipse import LsqEllipse
from matplotlib.patches import Ellipse

def cropping_images(mask, images):

    result = []
    t2_result = []
    adc_result = []
    bval_result = []
    ktrans_result = []
    for i in range(mask.shape[0]) : 
        contours = measure.find_contours(mask[i], 0.8)
        # let's find the right contour
        len_max = 0
        contour_max = 0
        for j in range(len(contours)):
            if len(contours[j]) > len_max : 
                contour_max = j
                len_max = len(contours[j])

        lsqe = LsqEllipse()
        try : 
            lsqe.fit(contours[contour_max])
            center, width, height, phi = lsqe.as_parameters()
            x_1 = int(center[0] - 64)
            x_2 = int(center[0] + 64)
            y_1 = int(center[1] + 64)
            y_2 = int(center[1] - 64)

            # cropping masks
            t2 = images[i, :, :, 0]
            adc = images[i, :, :, 1]
            bval = images[i, :, :, 2]
            ktrans = images[i, :, :, 3]

            zoomed_t2 = t2[x_1:x_2, y_2:y_1]
            zoomed_adc = adc[x_1:x_2, y_2:y_1]
            zoomed_bval = bval[x_1:x_2, y_2:y_1]
            zoomed_ktrans = ktrans[x_1:x_2, y_2:y_1]

            t2_result = t2_result + [zoomed_t2]
            adc_result = adc_result + [zoomed_adc]
            bval_result = bval_result + [zoomed_bval]
            ktrans_result = ktrans_result + [zoomed_ktrans]
        except : 
            print("no prostate", i)


    out_t2 = np.stack(t2_result, axis = 0)
    out_adc = np.stack(adc_result, axis = 0)
    out_bval = np.stack(bval_result, axis = 0)
    out_ktrans = np.stack(ktrans_result, axis = 0)

    out = np.stack([out_t2, out_adc, out_bval, out_ktrans], axis = -1)
    return out



print('Please enter the full path to the numpy file containing the MRI images')
print('Please also make sure the contained images are in the following order (1 image per dimension) : T2W, ADC, BVal, Ktrans')
path_to_nmpy = input() 

X_images = []
X_images = np.load(path_to_nmpy)

# load the models 
model_seg = get_model(256, 256, 1, "/Volumes/Lacie/Memoire/Backups/BackupPCLaboCOdesnweights/MasterThesisInfo2021/Promise12_Final/data/weights_with_augmentation.h5")
model_les = get_model_dc(128, 128, 4, "/Volumes/Lacie/Memoire/Backups/BackupPCLaboCOdesnweights/MasterThesisInfo2021/ProstateX_Final/Model/data/detection_and_classification/classifi_weights.h5")

f = open("Promise12_Final/normalization_values.txt", "r")
mu_seg = float(f.readline())
sigma_seg = float(f.readline())
f.close()

X_images_seg = (X_images[:, :, :, 0]-mu_seg)/sigma_seg


f = open("ProstateX_Final/Model/normalization_values.txt", "r")
mu_0 = float(f.readline())
sigma_0 = float(f.readline())

mu_1 = float(f.readline())
sigma_1 = float(f.readline())

mu_2 = float(f.readline())
sigma_2 = float(f.readline())

mu_3 = float(f.readline())
sigma_3 = float(f.readline())

f.close()

X_images_les = X_images
X_images_les[0] = (X_images[0]-mu_0)/sigma_0
X_images_les[1] = (X_images[1]-mu_1)/sigma_1
X_images_les[2] = (X_images[2]-mu_2)/sigma_2
X_images_les[3] = (X_images[3]-mu_3)/sigma_3



if X_images_seg[0].shape[1] != 256 and X_images_seg[0].shape[1] != 256 : 
    X_images_seg = resize_image(X_images_seg, 256, 256)
X_images_seg = np.expand_dims(X_images_seg, axis=3)
pred_seg = model_seg.predict(X_images_seg, verbose=1, batch_size=1)

for i in range(len(X_images_les)):
    cropped_0 = resize_image(X_images_les[:, :, :, 0], 256, 256)
    cropped_1 = resize_image(X_images_les[:, :, :, 1], 256, 256)
    cropped_2 = resize_image(X_images_les[:, :, :, 2], 256, 256)
    cropped_3 = resize_image(X_images_les[:, :, :, 3], 256, 256)
    out = np.stack([cropped_0, cropped_1, cropped_2, cropped_3], axis = -1)

cropped_images = cropping_images(pred_seg[:, :, :, 0], out)
X_les = np.expand_dims(cropped_images, axis=4)
pred_les = model_les.predict(X_les, verbose=1, batch_size=1)

myshow(sitk.GetImageFromArray(X_images_seg[:, :, :, 0]))
myshow(sitk.GetImageFromArray(pred_seg[:, :, :, 0]))
myshow(sitk.GetImageFromArray(X_les[:, :, :, 0]))
myshow(sitk.GetImageFromArray(pred_les[:, :, :, 1]))
myshow(sitk.GetImageFromArray(pred_les[:, :, :, 2]))

model_les.layers[-1].activation = None
img_3d_array = X_les
preds = model_les.predict(img_3d_array)
myshow(sitk.GetImageFromArray(preds[:, :, :, 0]))
