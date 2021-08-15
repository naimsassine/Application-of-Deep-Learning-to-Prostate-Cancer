"""
This code contains different functions to preprocess the masks and images for prostate and lesions
"""

import SimpleITK as sitk
import numpy as np
from pathlib import Path
import nibabel as nib
import os
from dipy.align.imaffine import AffineMap
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import SimpleITK as sitk
from pathlib import Path
import numpy as np
from skimage import measure

# Inspired by - https://github.com/alexhamiltonRN
def resample_voxel_spacing(is_training_data, desired_voxel_spacing):
    
    # Setting the paths

    path_to_nifti = Path()
    path_to_nifti_resampled = Path()
    counter = 1

    #path_to_nifti = Path('E:/Memoire/ProstateX/Masks/Files/prostate/mask_prostate')
    #path_to_nifti_resampled = Path('E:/Memoire/ProstateX/generated/Masks/Resampled/Prostate_Masks')

    path_to_nifti = Path('E:/Memoire/ProstateX/Masks/Files/lesions/Masks/T2')
    path_to_nifti_resampled = Path('E:/Memoire/ProstateX/generated/Masks/Resampled/T2_Lesions_Masks')
    
    # Methods

    def resample_image(desired_voxel_spacing, source_file_path):
        image = sitk.ReadImage(str(source_file_path))
        original_image_spacing = image.GetSpacing()

        if original_image_spacing != desired_voxel_spacing:
            ### inspired by : https://github.com/SimpleITK/SimpleITK/issues/561 ###
            
            # Converting to np array for calculations of new_size
            original_size_array = np.array(image.GetSize(), dtype = np.int)
            original_spac_array = np.array(image.GetSpacing())
            desired_spac_array = np.array(desired_voxel_spacing)
            
            new_size = original_size_array * (original_spac_array / desired_spac_array)
            new_size = np.ceil(new_size).astype(np.int)
            new_size = [int(s) for s in new_size]
            new_size = tuple(new_size)
            
            # Create the resample filter
            resample = sitk.ResampleImageFilter()
            resample.SetInterpolator(sitk.sitkLinear) 
            resample.SetSize(new_size)
            resample.SetOutputOrigin(image.GetOrigin()) 
            resample.SetOutputSpacing(desired_voxel_spacing)
            resample.SetOutputDirection(image.GetDirection())
            
            try:
                resampled_image = resample.Execute(image)
                # Print the changes
                print('\n')
                print('Resampling:', "/".join(source_file_path.parts[-5:]))
                print('original spacing:', image.GetSpacing())
                print('desired spacing:', desired_voxel_spacing)
                print('resampled spacing:', resampled_image.GetSpacing())
                print('original size:', image.GetSize())
                print('resampled size:', resampled_image.GetSize())
                print('\n')
            except:
                print('Problem with resampling image.')
                
        else:
            resampled_image = image
        
        return resampled_image
    
    def write_resampled_image(image, path, counter):
        writer = sitk.ImageFileWriter()
        writer.SetFileName(str(path))
        writer.Execute(image)
        print('Saving image to:', "/".join(path.parts[-5:]))
        counter = counter + 1
        return counter
        
    patient_files = [x for x in path_to_nifti.iterdir()]
    for file_path in patient_files:
        if "._" not in str(file_path) : 
            path_t2_resampled = path_to_nifti_resampled.joinpath("/".join(file_path.parts[-1:]))
            t2_resampled = resample_image(desired_voxel_spacing.get('t2'), file_path)
            counter = write_resampled_image(t2_resampled, path_t2_resampled,counter)             

    print('\n ++++ Files reviewed for resampling:', counter)

# Inspired by - https://github.com/alexhamiltonRN
def resample_spacing_main():
    is_training_data = False
    
    # Obtain input about datset from user
    dataset_type = input('Which dataset is being resampled? (1-Train; 2-Test):')
    if dataset_type == str(1):
        is_training_data = True
    
    # Set desired spacing based on EDA
    desired_voxel = {'t2':(0.5,0.5,3.0),
                     'adc':(2.0,2.0,3.0),
                     'bval':(2.0,2.0,3.0),
                     'ktrans':(1.5,1.5,4.0)} 
    
    resample_voxel_spacing(is_training_data, desired_voxel)

def save_rp_pm_nmpy() : 
    # register and flip prostate masks and save them
    t2_train_table = pd.read_csv('/Volumes/LaCie/Memoire/ProstateX/generated/train/dataframes/training_meta_data.csv')
    list_of_unique_t2 = []
    for i in range(len(t2_train_table)) :
        case = t2_train_table.iloc[i]
        if case["sequence_type"] == 't2': 
            list_of_unique_t2.append(case["resampled_nifti"])
    t2_paths = list(dict.fromkeys(list_of_unique_t2))

    path_to_pros_masks = 'E:/Memoire/ProstateX/generated/Masks/Resampled/Prostate_Masks/'

    for i in range(len(t2_paths)) :
        # inspired by https://towardsdatascience.com/3d-cnn-classification-of-prostate-tumour-on-multi-parametric-mri-sequences-prostatex-2-cced525394bb
        # for the registration part
        patient_id = t2_paths[i].split("/")[8][10:]
        pros_masks_path = path_to_pros_masks + "ProstateX-" + patient_id + ".nii.gz"
 
        t2 = nib.load(t2_paths[i])
        MASK_PROS = nib.load(pros_masks_path)

        t2_static = t2.get_fdata()
        t2_static_grid2world = t2.affine

        MASK_PROS_moving = MASK_PROS.get_fdata()
        MASK_PROS_moving_grid2world = MASK_PROS.affine

        identity = np.eye(4)
        MASK_PROS_affine_map = AffineMap(identity, t2_static.shape, t2_static_grid2world, MASK_PROS_moving.shape, MASK_PROS_moving_grid2world)

        transformMASK_PROS = MASK_PROS_affine_map.transform(MASK_PROS_moving)

        transposed_pros = transformMASK_PROS.transpose(2,0,1)
        for j in range(len(transposed_pros)) : 
            transposed_pros[j] = np.fliplr(np.rot90(np.fliplr(transposed_pros[j]), 3))
        filepath = pros_masks_path.replace("Resampled", "Registered_Flipped")
        np.save(Path(filepath), transposed_pros)

def save_rp_lm_nmpy() : 
    # register and flip lesion masks and save them
    t2_train_table = pd.read_csv('/Volumes/LaCie/Memoire/ProstateX/generated/train/dataframes/training_meta_data.csv')
    t2_paths = []
    for i in range(len(t2_train_table)) :
        case = t2_train_table.iloc[i]
        if case["sequence_type"] == 't2': 
            t2_paths.append(case["resampled_nifti"])

    copy = []
    for i in range(len(t2_paths)) : 
        if i not in [9, 11, 36, 37, 144, 154, 153, 194, 196, 206, 208, 209, 210, 225, 241, 243, 251, 253, 260, 263, 274, 275, 282, 258, 207, 133, 132, 35, 34, 68, 72] :
            copy.append(t2_paths[i])

    t2_paths = copy
    path_to_lesion_masks = 'E:/Memoire/ProstateX/generated/Masks/Resampled/T2_Lesions_Masks/'
    list_of_cases_done = ["0150", "0130"] # I already add this first case to create accordance with the dataset
    special_cases = ["0005", "0021", "0033", "0053", "0093", "0100", "0117", "0137", "0144", "0157", "0163", "0177", "0187", "0172"]
    not_available = ["0052", "0056", "0080", "0138"]
    for i in range(len(t2_paths)) :
        patient_id = t2_paths[i].split("/")[8][10:]
        list_of_cases_done.append(patient_id)
        if list_of_cases_done.count(patient_id) > 0 and patient_id not in special_cases : 
            finding_num = "-Finding" + str(list_of_cases_done.count(patient_id)) + "-"
        elif patient_id == "0005" and list_of_cases_done.count(patient_id) == 1 : 
            finding_num = "-Finding0-"
        elif patient_id == "0005" and list_of_cases_done.count(patient_id) == 2 : 
            finding_num = "-Finding1-"
        elif patient_id == "0021" and list_of_cases_done.count(patient_id) == 1: 
            finding_num = "-Finding2-"
        elif patient_id == "0021" and list_of_cases_done.count(patient_id) == 2: 
            finding_num = "-Finding4-"
        elif patient_id == "0033" and list_of_cases_done.count(patient_id) == 2: 
            finding_num = "-Finding3-"
        elif patient_id == "0053" and list_of_cases_done.count(patient_id) == 1: 
            finding_num = "-Finding4-"
        elif patient_id == "0093" and list_of_cases_done.count(patient_id) == 1: 
            finding_num = "-Finding3-"
        elif patient_id == "0100" and list_of_cases_done.count(patient_id) == 2: 
            finding_num = "-Finding3-"
        elif patient_id == "0117" and list_of_cases_done.count(patient_id) == 2: 
            finding_num = "-Finding3-"
        elif patient_id == "0137" and list_of_cases_done.count(patient_id) == 1: 
            finding_num = "-Finding3-"
        elif patient_id == "0144" and list_of_cases_done.count(patient_id) == 2: 
            finding_num = "-Finding3-"
        elif patient_id == "0157" and list_of_cases_done.count(patient_id) == 2: 
            finding_num = "-Finding3-"
        elif patient_id == "0163" and list_of_cases_done.count(patient_id) == 2: 
            finding_num = "-Finding3-"
        elif patient_id == "0177" and list_of_cases_done.count(patient_id) == 3: 
            finding_num = "-Finding4-"
        elif patient_id == "0187" and list_of_cases_done.count(patient_id) == 3: 
            finding_num = "-Finding4-"
        elif patient_id == "0172" and list_of_cases_done.count(patient_id) == 1: 
            finding_num = "-Finding4-"
        else : 
            finding_num = "-Finding1-"

        les_mask_path = path_to_lesion_masks + "ProstateX-" + patient_id + finding_num + "t2_tse_tra_ROI.nii.gz"



        if t2_paths[i].split("/")[8][10:] == les_mask_path.split("/")[9].split("-")[1] and patient_id not in not_available : 

            t2 = nib.load(t2_paths[i])
            try : 
                MASK_LES = nib.load(les_mask_path)
            except : 
                les_mask_path = path_to_lesion_masks + "ProstateX-" + patient_id + finding_num + "t2_tse_tra0_ROI.nii.gz"
                MASK_LES = nib.load(les_mask_path)
            # inspired by https://towardsdatascience.com/3d-cnn-classification-of-prostate-tumour-on-multi-parametric-mri-sequences-prostatex-2-cced525394bb
            # for the registration part
            t2_static = t2.get_fdata()
            t2_static_grid2world = t2.affine

            MASK_LES_moving = MASK_LES.get_fdata()
            MASK_LES_moving_grid2world = MASK_LES.affine

            identity = np.eye(4)
            MASK_LES_affine_map = AffineMap(identity, t2_static.shape, t2_static_grid2world, MASK_LES_moving.shape, MASK_LES_moving_grid2world)

            transformMASK_LES = MASK_LES_affine_map.transform(MASK_LES_moving)

            transposed_les = transformMASK_LES.transpose(2,0,1)
            for j in range(len(transposed_les)) : 
                transposed_les[j] = np.fliplr(np.rot90(np.fliplr(transposed_les[j]), 3))
            filepath = les_mask_path.replace("Resampled", "Registered_Flipped")
            np.save(Path(filepath), transposed_les)
        else : 
            print("wrong patient")
            print(patient_id, "\n")

def sum_lesions_to_pros_masks_beta() :
    # we have the significant + mask apart from the none-significant + mask. So here I am summing eveyrthing
    path_pros = '/Volumes/LaCie/Memoire/ProstateX/generated/Masks/Registered_Flipped/Prostate_Masks/'
    path_les = '/Volumes/LaCie/Memoire/ProstateX/generated/Masks/Registered_Flipped/T2_Lesions_Masks/'
    lesions_table = pd.read_csv('/Volumes/LaCie/Memoire/ProstateX/Masks/Files/lesions/PROSTATEx_Classes.csv')
    for i in range(0, 299) : 
        lesion_case = lesions_table.iloc[i]
        if "0025" in lesion_case["ID"][:14]: 
            print("Unauthorized Value")
        else : 
            prostate_mask = np.load(path_pros + lesion_case["ID"][:14] + ".nii.gz.npy")
            try : 
                lesion_mask = np.load(path_les + lesion_case["ID"].replace("_", "-") + "-t2_tse_tra_ROI.nii.gz.npy")
            except : 
                lesion_mask = np.load(path_les + lesion_case["ID"].replace("_", "-") + "-t2_tse_tra0_ROI.nii.gz.npy")

            thresh = 0.0
            maxval = 1
            bin_prostate = (prostate_mask > thresh) * maxval
            bin_lesion = (lesion_mask > thresh) * maxval

            if lesion_case["Clinically Significant"] == True : 
                summed = bin_prostate + bin_lesion + bin_lesion
                summed = np.where(summed==2, 3, summed) 
            else : 
                summed = bin_prostate + bin_lesion
            
            filepath = 'E:/Memoire/ProstateX/generated/Masks/Summed/' + lesion_case["ID"]
            np.save(Path(filepath), summed)

def sum_lesions_to_pros_masks_final():
    # for each patient
    # we fetch the prostate mask
    # we then fetch all the findings and store them in a list using the significance function
    # in this list we also have a True/False value to indicate if the lesion is significant or not
    # for each finding, we fetch it, if its true, we sum it 6 times over itself, and if its false, 4 times
    # we do that for one reason : to differenciate the pixels that indicate a significat lesion and a none-sign
    # we sum the findings over themselves, so that if a patient has none-sign and sign findings, we'll have them in the same file
    # and final step we sum the prostate mask over all of that. We then apply a reduction function that transforms the pixel
    # values to the one we want for deep learning : 
    # 0 for the background
    # 1 for the prostate
    # 2 for the lesions non sign
    # 3 for the lesions sign

    path_pros_mask = 'E:/Memoire/ProstateX/generated/Masks/Registered_Flipped/Prostate_Masks/'
    path_lesion_mask = 'E:/Memoire/ProstateX/generated/Masks/Registered_Flipped/T2_Lesions_Masks/'
    not_allowed = [25, 38]
    for i in range(0, 204) :
        if i not in not_allowed : 
            patient_id = 0
            if i < 10 :
                patient_id = "000" + str(i)
            elif i < 100 :
                patient_id = "00" + str(i)
            else :
                patient_id = "0" + str(i)
            pros_mask = np.load(path_pros_mask + "ProstateX-" + patient_id + ".nii.gz.npy")
            findings = significance(patient_id)
            summed = 0
            thresh = 0.0
            maxval = 1
            bin_prostate = (pros_mask > thresh) * maxval
            for finding in findings : 
                try : 
                    lesion_mask = np.load(path_lesion_mask + "ProstateX-" + patient_id + "-Finding" + finding  + "-t2_tse_tra_ROI.nii.gz.npy")
                except : 
                    lesion_mask = np.load(path_lesion_mask + "ProstateX-" + patient_id + "-Finding" + finding  + "-t2_tse_tra0_ROI.nii.gz.npy")


                bin_lesion = (lesion_mask > thresh) * maxval
                
                if findings[finding] == True : 
                    summed =  summed + bin_lesion + bin_lesion + bin_lesion + bin_lesion + bin_lesion + bin_lesion 
                else : 
                    summed =  summed + bin_lesion + bin_lesion + bin_lesion + bin_lesion
                # in case of superposition
                summed = np.where(summed==10, 6, summed)
                summed = np.where(summed==8, 4, summed)
                summed = np.where(summed==12, 6, summed)
                summed = np.where(summed==14, 6, summed)
            
            summed = summed + bin_prostate
            summed = np.where(summed==7, 3, summed)
            summed = np.where(summed==6, 3, summed)
            summed = np.where(summed==4, 2, summed)
            summed = np.where(summed==5, 2, summed)
            filepath = 'E:/Memoire/ProstateX/generated/mutli_class_segm/train_masks_not_cropped_in_one/' + patient_id
            np.save(Path(filepath), summed)
        else : 
            print("Unauthorized value")

def significance(patient_id) :
    # returns the fids, with significance 
    # summed = np.where(summed==2, 3, summed) 
    lesions_table = pd.read_csv('E:/Memoire/ProstateX/Masks/Files/lesions/PROSTATEx_Classes.csv')
    locations = []
    dico = {}
    for i in range(0, 299) :
        lesion_case = lesions_table.iloc[i]
        if lesion_case["ID"][:14][10:] == patient_id :
            locations.append(i)
    
    for loc in locations :
        finding_case = lesions_table.iloc[loc]
        fid = finding_case["ID"][-1]
        dico[fid] = finding_case["Clinically Significant"]

    return dico

# method that registers the t2, adc, bval and ktrans images and stores them into a numpy of size (z, 384, 384, 4)
# the registration code is inspired by https://towardsdatascience.com/3d-cnn-classification-of-prostate-tumour-on-multi-parametric-mri-sequences-prostatex-2-cced525394bb
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
    # given a t2 location, retrieve the location of the other types of MRIs for this specific patient
    t2_train_table = pd.read_csv('/Volumes/LaCie/Memoire/ProstateX/generated/train/dataframes/training_meta_data.csv')
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

from ellipse import LsqEllipse
from matplotlib.patches import Ellipse

def cropping_images():
    # let's crop the images depending on the prostate mask
    path = Path()
    path = Path('E:/Memoire/ProstateX/generated/mutli_class_segm/train_masks_not_cropped_in_one/')

    patient_files = [x for x in path.iterdir()]
    for file_path in patient_files:
        mask = np.load(file_path)
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
            except : 
                center = (64, 64)


            x_1 = int(center[0] - 64)
            x_2 = int(center[0] + 64)
            y_1 = int(center[1] + 64)
            y_2 = int(center[1] - 64)

            # cropping masks
            zoomed = mask[i][x_1:x_2, y_2:y_1]
            result = result + [zoomed]

            # cropping training images
            path_to_training = 'E:/Memoire/ProstateX/generated/mutli_class_segm/train_images_not_cropped/' + 'ProstateX-' + str(file_path).split("/")[-1]
            training_image = np.load(path_to_training)
            t2 = training_image[i, :, :, 0]
            adc = training_image[i, :, :, 1]
            bval = training_image[i, :, :, 2]
            ktrans = training_image[i, :, :, 3]

            zoomed_t2 = t2[x_1:x_2, y_2:y_1]
            zoomed_adc = adc[x_1:x_2, y_2:y_1]
            zoomed_bval = bval[x_1:x_2, y_2:y_1]
            zoomed_ktrans = ktrans[x_1:x_2, y_2:y_1]

            t2_result = t2_result + [zoomed_t2]
            adc_result = adc_result + [zoomed_adc]
            bval_result = bval_result + [zoomed_bval]
            ktrans_result = ktrans_result + [zoomed_ktrans]
    

        saved_path = 'E:/Memoire/ProstateX/generated/mutli_class_segm/train_masks_cropped_in_one/' + str(file_path).split("/")[-1]
        out = np.stack(result, axis = 0)
        np.save(Path(saved_path), out)

        saved_path = 'E:/Memoire/ProstateX/generated/mutli_class_segm/train_images_cropped/' + str(file_path).split("/")[-1]
        out_t2 = np.stack(t2_result, axis = 0)
        out_adc = np.stack(adc_result, axis = 0)
        out_bval = np.stack(bval_result, axis = 0)
        out_ktrans = np.stack(ktrans_result, axis = 0)

        out = np.stack([out_t2, out_adc, out_bval, out_ktrans], axis = -1)
        np.save(Path(saved_path), out)

def concatenate_train_images(cases):
    # let's concatenate everything into a single numpy
    # takes is as input the output of the next function below
    path = Path()
    path = Path('E:/Memoire/ProstateX/generated/mutli_class_segm/train_images_cropped/')

    patient_files = [x for x in path.iterdir()]
    result = []
    for file_path in sorted(patient_files):
        if str(file_path).split("\\")[6][:4] in cases : 
            image_3d = np.load(file_path)
            slices = cases[str(file_path).split("\\")[6][:4]]
            for s in slices : 
                result = result + [[image_3d[s]]]
    
    concatenated = np.concatenate(result, 0)
    print(concatenated.shape)
    np.save('E:/Memoire/ProstateX/generated/mutli_class_segm/train_images_concatenated/train_images_concatenated.npy', concatenated)

def concatenate_train_masks():
    # let's concatenate everything into a single numpy
    path_masks = Path()
    path_masks = Path('E:/Memoire/ProstateX/generated/mutli_class_segm/train_masks_cropped_in_one/')

    patient_files = [x for x in path_masks.iterdir()]
    result = []
    cases = {}
    for file_path in sorted(patient_files):
        volume = np.load(file_path)
        if len(np.unique(volume)) > 2:
            slices = detect_lesions_in_volume(volume)
            cases[str(file_path).split("\\")[6][:4]] = slices
            for s in slices:
                result = result + [[volume[s]]]

    
    concatenated = np.concatenate(result, 0)
    print(concatenated.shape)
    np.save('E:/Memoire/ProstateX/generated/mutli_class_segm/train_masks_concatenated/train_masks_concatenated.npy', concatenated)
    return cases

def detect_lesions_in_volume(volume):
    # here I am only choosing the slices of each patient's volume where there is a prostate
    # I do that because in a volume of let's say 26 MRIS, only half of them contain a clear view of the prostate
    # the other half only contains the surroundings. Since the first model of this master thesis indicates where there is a prostate
    # and where there isnt', it makes no sense to train the model on images where we don't have a prostate. This makes training much
    # more easy. So in the training set of this escond model, we have images containing cancers, and images not containing cancers, but we'll always
    # have images containing a prostate.
    slices = [] 
    for i in range(volume.shape[0]):
        if len(np.unique(volume[i])) > 2 : 
            slices.append(i)
    slices_2 = []
    for i in slices: 
        for j in range(0, volume.shape[0], 3):
            if j not in slices and j not in slices_2 and len(np.unique(volume[j]))>1: 
                slices_2.append(j)
                break
    return slices + slices_2

