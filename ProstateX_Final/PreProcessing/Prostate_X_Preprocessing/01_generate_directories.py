"""
Taken from - https://github.com/alexhamiltonRN
"""
from pathlib import Path

def generate_patient_ids(dataset_type):
    """
    This function generates the patient_ids for the directories to be created below. 
    Ids are extracted from the raw dataset file structure.
    """
    
    patient_ids = []
    path_to_date = Path()
    
    if dataset_type == str(1):
        path_to_data = Path('E:/Memoire/ProstateX/train-data')
    else:
        path_to_data = Path('E:/Memoire/ProstateX/test-data')
    
    # Get list of patient_ids in folder
    patient_folders = [x for x in path_to_data.iterdir() if x.is_dir()]
    for patient_folder in patient_folders:
        patient_ids.append(str(patient_folder.stem))
    return patient_ids 

def generate_nifti_ds(patient_ids, dataset_type):
    """
    This function generates the directory structure for the nifti files
    generated from the dicom files.

    Directory structure for generated data:
    ProstateX/generated/train/nifti
    ProstateX/generated/test/nifti
    """
    for patient_id in patient_ids:
        if dataset_type == str(1):
            new_path = Path(str('E:/Memoire/ProstateX/generated/train/nifti/' + patient_id))
            new_path.mkdir(parents = True, exist_ok = True)
            new_path.joinpath('t2').mkdir(parents = True, exist_ok = True)
            new_path.joinpath('bval').mkdir(parents = True, exist_ok = True)
            new_path.joinpath('adc').mkdir(parents = True, exist_ok = True)
            new_path.joinpath('ktrans').mkdir(parents = True, exist_ok = True)

        else:
            new_path = Path(str('E:/Memoire/ProstateX/generated/test/nifti/' + patient_id))
            new_path.mkdir(parents = True, exist_ok = True)
            new_path.joinpath('t2').mkdir(parents = True, exist_ok = True)
            new_path.joinpath('bval').mkdir(parents = True, exist_ok = True)
            new_path.joinpath('adc').mkdir(parents = True, exist_ok = True)
            new_path.joinpath('ktrans').mkdir(parents = True, exist_ok = True)

def generate_nifti_resampled_ds(patient_ids, dataset_type):
    """
    This function generates the directory structure for the nifti files
    generated from the dicom files.

    Directory structure for generated data:
    ProstateX/generated/train/nifti_resampled
    ProstateX/generated/test/nifti_resampled
    """
    for patient_id in patient_ids:
        if dataset_type == str(1):
            new_path = Path(str('E:/Memoire/ProstateX/generated/train/nifti_resampled/' + patient_id))
            new_path.mkdir(parents = True, exist_ok = True)
            new_path.joinpath('t2').mkdir(parents = True, exist_ok = True)
            new_path.joinpath('bval').mkdir(parents = True, exist_ok = True)
            new_path.joinpath('adc').mkdir(parents = True, exist_ok = True)
            new_path.joinpath('ktrans').mkdir(parents = True, exist_ok = True)

        else:
            new_path = Path(str('E:/Memoire/ProstateX/generated/test/nifti_resampled/' + patient_id))
            new_path.mkdir(parents = True, exist_ok = True)
            new_path.joinpath('t2').mkdir(parents = True, exist_ok = True)
            new_path.joinpath('bval').mkdir(parents = True, exist_ok = True)
            new_path.joinpath('adc').mkdir(parents = True, exist_ok = True)
            new_path.joinpath('ktrans').mkdir(parents = True, exist_ok = True)

def generate_numpy_ds(dataset_type):
    """
    This function generates the directory structure for the final numpy
    arrays for the training and test sets. 
    
    Director structure for processed data:
    ProstateX/generated/train/numpy
    ProstateX/generated/test/numpy
    """
    if dataset_type == str(1):
        new_path = Path('E:/Memoire/ProstateX/generated/train/numpy/')
        new_path.mkdir(parents = True, exist_ok = True)
        new_path.joinpath('t2').mkdir(parents = True, exist_ok = True)
        new_path.joinpath('bval').mkdir(parents = True, exist_ok = True)
        new_path.joinpath('adc').mkdir(parents = True, exist_ok = True)
        new_path.joinpath('ktrans').mkdir(parents = True, exist_ok = True)
    else:
        new_path = Path('E:/Memoire/ProstateX/generated/test/numpy/')
        new_path.mkdir(parents = True, exist_ok = True)
        new_path.joinpath('t2').mkdir(parents = True, exist_ok = True)
        new_path.joinpath('bval').mkdir(parents = True, exist_ok = True)
        new_path.joinpath('adc').mkdir(parents = True, exist_ok = True)
        new_path.joinpath('ktrans').mkdir(parents = True, exist_ok = True)
        
def generate_dataframe_ds(dataset_type):
    if dataset_type == str(1):
        new_path = Path('E:/Memoire/ProstateX/generated/train/dataframes/')
        new_path.mkdir(parents = True, exist_ok = True)

    else:
        new_path = Path('E:/Memoire/ProstateX/generated/test/dataframes/')
        new_path.mkdir(parents = True, exist_ok = True)

def generate_logs_ds(dataset_type):
    if dataset_type == str(1):
        new_path = Path('E:/Memoire/ProstateX/generated/train/logs/')
        new_path.mkdir(parents = True, exist_ok = True)

    else:
        new_path = Path('E:/Memoire/ProstateX/generated/test/logs/')
        new_path.mkdir(parents = True, exist_ok = True)

def main():
    dataset_type = input('Generate directory structure for which type of data (1-Train; 2-Test):')
    patient_ids = generate_patient_ids(dataset_type)
    generate_nifti_ds(patient_ids, dataset_type)
    generate_nifti_resampled_ds(patient_ids, dataset_type)
    generate_numpy_ds(dataset_type)
    generate_dataframe_ds(dataset_type)
    generate_logs_ds(dataset_type)
    print('Done creating directory structure...')

main()