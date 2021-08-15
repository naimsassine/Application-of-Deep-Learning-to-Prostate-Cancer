"""
Taken from - https://github.com/alexhamiltonRN
"""

import pandas as pd
import dicom2nifti
import SimpleITK as sitk
import os
from pathlib import Path

def generate_paths_to_dicom(is_training_data):
    """
    This function generates a dictionary containing the patient id as the 
    primary key and assigns a second dictionary containing the paths to the 
    t2, adc, and bval dicom folders for conversion to nifti later
    """
    
    path_to_data = Path()
    paths_to_dicom = {}
    
    if is_training_data:
        path_to_data = Path('E:/Memoire/ProstateX/train-data')
    else:
        path_to_data = Path('E:/Memoire/ProstateX/test-data') 
    
    patient_folders = [x for x in path_to_data.iterdir() if x.is_dir()]
    for patient_folder in patient_folders:
        patient_path = patient_folder.stem
        subdirectories = [x for x in patient_folder.iterdir() if x.is_dir()]
        for subdirectory in subdirectories:
            t2_path = Path()
            adc_path = Path()
            bval_path = Path()

            scan_folders_paths = [x for x in subdirectory.iterdir() if x.is_dir()]
            
            for folder in scan_folders_paths: 
                if 't2tsetra' in str(folder):
                    t2_path = folder
                if 'ADC' in str(folder):
                    adc_path = folder
                if 'BVAL' in str(folder):
                    bval_path = folder
            
            paths_to_dicom[patient_path] = {'t2':t2_path, 'adc': adc_path, 'bval':bval_path} 
    
    print('Done generating paths to dicom files for conversion...\n')
    return paths_to_dicom

def convert_dicom2nifti(paths_to_dicom, is_training_data):
    """
    This function does the actual conversion of dicom files to nifti by 
    supplying the dicom2nifti convert_directory method with a path to the 
    original dicom files and to a new directory.
    """
    print('Generating nifti files from dicom...\n')
    
    path_to_nifti = Path()

    if is_training_data:
        path_to_nifti = Path('E:/Memoire/ProstateX/generated/train/nifti/')
    else:
        path_to_nifti = Path('E:/Memoire/ProstateX/generated/test/nifti/')

    counter = 1

    for patient_id, file_structure in paths_to_dicom.items():
         
        dicom_t2_path = file_structure['t2']
        dicom_adc_path = file_structure['adc']
        dicom_bval_path = file_structure['bval']

        nifti_t2_path = path_to_nifti.joinpath(str(patient_id) + '/t2')
        nifti_adc_path = path_to_nifti.joinpath(str(patient_id) + '/adc')
        nifti_bval_path = path_to_nifti.joinpath(str(patient_id) + '/bval')

        try:
            dicom2nifti.convert_directory(str(dicom_t2_path), str(nifti_t2_path))
            dicom2nifti.convert_directory(str(dicom_adc_path), str(nifti_adc_path))
            dicom2nifti.convert_directory(str(dicom_bval_path), str(nifti_bval_path))
            print('Successful dicom to nifti conversion of: ' + patient_id, counter)
            counter = counter + 1

        except:
            if is_training_data:
                f = open('E:/Memoire/ProstateX/generated/train/dicom2nifti_train_problem_cases.txt', 'a+')
                f.write(patient_id +"\n")
                f.close()
                print('Problem with:', patient_id)
            else:
                f = open('E:/Memoire/ProstateX/generated/test/dicom2nifti_train_problem_cases.txt', 'a+')
                f.write(patient_id +"\n")
                f.close()
                print('Problem with:', patient_id)
            continue
   
def convert_mhd2nifti(is_training_data):
    """
    This function converts mhd (ktrans) files to nifti. It uses SimpleITK ImageFileReader and 
    ImageFileWriter functions to execute the conversion.
    """
    
    print('\n')
    print('Generating nifti files from mhd...\n')

    if is_training_data:
        path_to_nifti = Path('E:/Memoire/ProstateX/generated/train/nifti/')
        path_to_ktrans_data = Path('E:/Memoire/ProstateX/other/ktrans-train')
    else:
        path_to_nifti = Path('E:/Memoire/ProstateX/generated/test/nifti')
        path_to_ktrans_data = Path('E:/Memoire/ProstateX/other/ktrans-test')

    counter = 1
    patient_folders = [x for x in path_to_ktrans_data.iterdir() if x.is_dir()]

    for patient in patient_folders:
        patient_id = patient.stem

        mhd_files = []
        
        for item in patient.rglob('*.mhd'):
            mhd_files.append(item)

        full_path_to_mhdfile = mhd_files[0]
        full_path_to_ktrans_folder = path_to_nifti / patient_id / 'ktrans'
        filename = patient_id + '-ktrans.nii.gz' 
        new_file_path = full_path_to_ktrans_folder / filename
        
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(full_path_to_mhdfile))
        image = reader.Execute()

        writer = sitk.ImageFileWriter()
        writer.SetFileName(str(new_file_path))
        writer.Execute(image)    

        print('Successful mhd to nifti conversion of: ' + patient_id, counter)
        counter = counter + 1

def main():
    is_training_data = False
    
    dataset_type = input('What type of data to convert? (1-Train; 2-Test):')
    if dataset_type == str(1):
        is_training_data = True
    
    paths_to_dicom = generate_paths_to_dicom(is_training_data)
    convert_dicom2nifti(paths_to_dicom, is_training_data)
    convert_mhd2nifti(is_training_data)
    
main()