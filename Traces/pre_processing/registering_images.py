import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import SimpleITK as sitk
from pathlib import Path
from myshow import myshow, myshow3d
import numpy as np
import nibabel as nib
import os
from dipy.align.imaffine import AffineMap


# method that registers the t2, adc, bval and ktrans images and stores them into a numpy of size (z, 384, 384, 4)
def register_images() : 

    t2_train_table = pd.read_csv('E:/Memoire/ProstateX/generated/train/dataframes/training_meta_data.csv')

    for i in range(0, 330) : 
        t2_case = t2_train_table.iloc[i]
        adc_location, bval_location, ktrans_location = search_for_images(i)
        adc = t2_train_table.iloc[adc_location]
        bval = t2_train_table.iloc[bval_location]
        ktrans = t2_train_table.iloc[ktrans_location]
        if t2_case["ProxID"] == adc["ProxID"] == bval["ProxID"] == ktrans["ProxID"] : 

            t2 = nib.load(str(t2_case['resampled_nifti']))
            t2_image = sitk.ReadImage(str(t2_case['resampled_nifti']))
            ADC = nib.load(str(adc['resampled_nifti']))
            BVAL = nib.load(str(bval['resampled_nifti']))
            KTRANS = nib.load(str(ktrans['resampled_nifti']))

            t2_static = t2.get_fdata()
            t2_static_grid2world = t2.affine

            ADC_moving = ADC.get_fdata()
            ADC_moving_grid2world = ADC.affine

            identity = np.eye(4)
            ADC_affine_map = AffineMap(identity, t2_static.shape, t2_static_grid2world, ADC_moving.shape, ADC_moving_grid2world)

            transformADC = ADC_affine_map.transform(ADC_moving)


            BVAL_moving = BVAL.get_fdata()
            BVAL_moving_grid2world = BVAL.affine

            identity = np.eye(4)
            BVAL_affine_map = AffineMap(identity, t2_static.shape, t2_static_grid2world, BVAL_moving.shape, BVAL_moving_grid2world)

            transformBVAL = BVAL_affine_map.transform(BVAL_moving)


            KTRANS_moving = KTRANS.get_fdata()
            KTRANS_moving_grid2world = KTRANS.affine

            identity = np.eye(4)
            KTRANS_affine_map = AffineMap(identity, t2_static.shape, t2_static_grid2world, KTRANS_moving.shape, KTRANS_moving_grid2world)

            transformKTRANS = KTRANS_affine_map.transform(KTRANS_moving)


            transposed_adc = transformADC.transpose(2,0,1)
            for j in range(len(transposed_adc)) : 
                transposed_adc[j] = np.fliplr(np.flipud(np.rot90(np.fliplr(transposed_adc[j]), 3)))

            transposed_bval = transformBVAL.transpose(2,0,1)
            for j in range(len(transposed_bval)) : 
                transposed_bval[j] = np.fliplr(np.flipud(np.rot90(np.fliplr(transposed_bval[j]), 3)))

            transposed_ktrans = transformKTRANS.transpose(2,0,1)
            for j in range(len(transposed_ktrans)) : 
                transposed_ktrans[j] = np.fliplr(np.rot90(np.fliplr(transposed_ktrans[j]), 3))

            out = np.stack([sitk.GetArrayViewFromImage(t2_image) , transposed_adc, transposed_bval, transposed_ktrans], axis = -1)
            path = 'E:/Memoire/ProstateX/generated/mutli_class_segm/train_images_not_cropped/' + t2_case["ProxID"]
            np.save(path, out)
        else : 
            print("Not the same patient")
            print(t2_case["ProxID"]) 
            print(adc["ProxID"]) 
            print(bval["ProxID"]) 
            print(ktrans["ProxID"])



def search_for_images(location) :
    t2_train_table = pd.read_csv('E:/Memoire/ProstateX/generated/train/dataframes/training_meta_data.csv')
    t2_case = t2_train_table.iloc[location]
    id = t2_case["ProxID"]

    adc_location = 0
    bval_location = 0
    ktrans_location = 0
    for i in range(330, 657) : 
        # search for adc location
        adc_case = t2_train_table.iloc[i]
        if adc_case["ProxID"] == id : 
            adc_location = i
    for i in range(657, 987) : 
        # search for bval location
        bval_case = t2_train_table.iloc[i]
        if bval_case["ProxID"] == id : 
            bval_location = i
    for i in range(987, 1317) : 
        # search for ktrans location
        ktrans_case = t2_train_table.iloc[i]
        if ktrans_case["ProxID"] == id : 
            ktrans_location = i

    return [adc_location, bval_location, ktrans_location]

register_images()