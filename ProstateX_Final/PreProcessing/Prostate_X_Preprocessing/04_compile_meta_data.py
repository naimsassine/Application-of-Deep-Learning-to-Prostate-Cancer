"""
Taken from - https://github.com/alexhamiltonRN
"""

import pandas as pd
import pickle
from pathlib import Path

def generate_cases_meta_df(is_training_data, sequence_type):
    """
    This function generates a data frame containing the necessary information (ProxID, DCMSerDesc,
    and path to resampled NIFTI file) for cases so that they can be joined to tabular information 
    provided by the research team. Data that will be merged with dataset are found in ProstateX-Images
    and ProstateX-Images-KTrans files (Train and Test Respectively) 
    """

    if is_training_data:
        path_lesion_information = Path('E:/Memoire/ProstateX/other/TrainingLesionInformation')
        path_resampled_nifti = Path('E:/Memoire/ProstateX/generated/train/nifti_resampled') 
    else:
        path_lesion_information = Path('E:/Memoire/ProstateX/other/TestLesionInformation')
        path_resampled_nifti = Path('E:/Memoire/ProstateX/generated/test/nifti_resampled')
    
    patient_data = {}

    def generate_DCMSerDescr_from_filename(item):
        # remove extension from path
        full_name = item.parts[-1]
        split = full_name.split('.')
        name_without_extension = split[0]

        # remove first num and underscore from path
        first_underscore = name_without_extension.find('_') + 1
        value = name_without_extension[first_underscore:]
        return value

    patient_folders = [x for x in path_resampled_nifti.iterdir() if x.is_dir()]
    for patient in patient_folders:
        sequences = [x for x in patient.iterdir() if x.is_dir()]
        for sequence in sequences:
            if sequence.parts[-1] == sequence_type:
                for item in sequence.rglob('*.*'):
                    constructed_DCMSerDescr = generate_DCMSerDescr_from_filename(item)
                    path_to_resampled = item

                    if 't2' in constructed_DCMSerDescr:
                        sequence_type = 't2'
                    elif 'adc' in constructed_DCMSerDescr:
                        sequence_type = 'adc'
                    elif 'bval' in constructed_DCMSerDescr:
                        sequence_type = 'bval'
                    else: 
                        sequence_type = 'ktrans'

                    key = patient.parts[-1]
                    value = [constructed_DCMSerDescr, path_to_resampled, sequence_type]
                    patient_data[key] = value

    cases_meta_data_df = pd.DataFrame.from_dict(patient_data, orient = 'index')
    cases_meta_data_df = cases_meta_data_df.reset_index()
    cases_meta_data_df.columns = ['ProxID', 'DCMSerDescr', 'resampled_nifti', 'sequence_type']
    
    return cases_meta_data_df


def join_data(is_training_data, sequence_df_array):
    """
    This function combines information provided by the research team in ProstateX-Images
    and ProstateX-Images-KTrans (Train/Test) files with paths to the resampled NIFTI files. The
    function accepts a boolean is_training_data to determine if it is training or test
    data that needs to be processed. A list containing data frames of the joined data
    is the second parameter. The function concatenates the data frames in this list and
    returns a final data frame of all the data.
    """

    if is_training_data:
        prostateX_images = pd.read_csv('E:/Memoire/ProstateX/other/TrainingLesionInformation/ProstateX-Images-Train.csv')
        prostateX_images_ktrans = pd.read_csv('E:/Memoire/ProstateX/other/TrainingLesionInformation/ProstateX-Images-KTrans-Train.csv')
        prostateX_findings = pd.read_csv('E:/Memoire/ProstateX/other/TrainingLesionInformation/ProstateX-Findings-Train.csv')
    else:
        prostateX_images = pd.read_csv('E:/Memoire/ProstateX/other/TestLesionInformation/ProstateX-Images-Test.csv')
        prostateX_images_ktrans = pd.read_csv('E:/Memoire/ProstateX/other/TestLesionInformation/ProstateX-Images-KTrans-Test.csv')
        prostateX_findings = pd.read_csv('E:/Memoire/ProstateX/other/TestLesionInformation/ProstateX-Findings-Test.csv')
  
    df_collection = []
    
    # Merging info for the DICOM series
    for dataframe in sequence_df_array[0:3]:
        # Convert DCMSerDescr values to lowercase in both frames (sanitize)
        dataframe.loc[:,'DCMSerDescr'] = dataframe.loc[:,'DCMSerDescr'].apply(lambda x: x.lower())
        prostateX_images.loc[:,'DCMSerDescr'] = prostateX_images.loc[:,'DCMSerDescr'].apply(lambda x: x.lower())
        
        # Keep only important columns from researcher provided data
        prostateX_images = prostateX_images[['ProxID', 'DCMSerDescr', 'fid', 'pos','WorldMatrix', 'ijk']]

        # Merge NIFTI paths with researcher provided data
        first_merge = pd.merge(dataframe, prostateX_images, how = 'inner', on = ['ProxID', 'DCMSerDescr'])
        
        # Merge findings (cancer/not cancer)
        final_merge = pd.merge(first_merge, prostateX_findings, how = 'inner', on = ['ProxID', 'fid', 'pos'])
        df_collection.append(final_merge)
    
   
    # Merging info for the KTRANS series
    first_merge = pd.merge(sequence_df_array[3], prostateX_images_ktrans, how = 'inner', on = ['ProxID'])
    
    # Merge findings (cancer/not cancer)
    final_merge = pd.merge(first_merge, prostateX_findings, how = 'inner', on = ['ProxID', 'fid', 'pos'])
    df_collection.append(final_merge)

    final_dataframe = pd.concat(df_collection, ignore_index=True)

    return final_dataframe

def repair_values(is_training_data, dataframe):
    """
    This function accepts a data frame and reformats entries in select columns
    to make them more acceptable for use in patch analysis (i.e. converting strings of 
    coordinate values to tuples of float).
    """

    def convert_to_tuple(dataframe, column):
        """
        This function converts row values (represented as string of floats
        delimited by spaces) to a tuple of floats. It accepts the original data
        frame and a string for the specified column that needs to be converted.
        """  
        pd_series_containing_lists_of_strings = dataframe[column].str.split() 
        list_for_new_series = []
        for list_of_strings in pd_series_containing_lists_of_strings:
            container_list = []
            for item in list_of_strings:
                if column == 'pos':
                    container_list.append(float(item))
                else:
                    container_list.append(int(item))
            list_for_new_series.append(tuple(container_list))
        
        return pd.Series(list_for_new_series)    

    # Call function to convert select columns
    dataframe = dataframe.assign(pos_tuple = convert_to_tuple(dataframe, 'pos'))
    dataframe = dataframe.assign(ijk_tuple = convert_to_tuple(dataframe, 'ijk'))
    
    # Drop old columns, rename new ones, and reorder...
    dataframe = dataframe.drop(columns = ['pos','ijk', 'WorldMatrix'])
    dataframe = dataframe.rename(columns = {'pos_tuple':'pos', 'ijk_tuple':'ijk'})

    if is_training_data:
        repaired_df = dataframe.loc[:,['ProxID', 'DCMSerDescr', 'resampled_nifti', 'sequence_type', 'fid', 'pos', 'ijk', 'zone', 'ClinSig']]
    else:
        repaired_df = dataframe.loc[:,['ProxID', 'DCMSerDescr', 'resampled_nifti', 'sequence_type', 'fid', 'pos', 'ijk', 'zone']]
    
    return repaired_df


def save_data_to_directory(is_training_data, dataframe):
    if is_training_data:
        dataframe.to_csv('E:/Memoire/ProstateX/generated/train/dataframes/training_meta_data.csv')
        dataframe.to_pickle('E:/Memoire/ProstateX/generated/train/dataframes/training_meta_data.pkl')
    else:
        dataframe.to_csv('E:/Memoire/ProstateX/generated/test/dataframes/test_meta_data.csv')
        dataframe.to_pickle('E:/Memoire/ProstateX/generated/test/dataframes/test_meta_data.pkl')

def main():
    is_training_data = False
    dataset_type = input('Which dataset are you working with? (1-Train; 2-Test):')
    if dataset_type == str(1):
        is_training_data = True
    
    t2_meta = generate_cases_meta_df(is_training_data, 't2')
    adc_meta = generate_cases_meta_df(is_training_data, 'adc')
    bval_meta = generate_cases_meta_df(is_training_data, 'bval')
    ktrans_meta = generate_cases_meta_df(is_training_data, 'ktrans')

    sequence_df_array = [t2_meta, adc_meta, bval_meta, ktrans_meta]
    complete_df = join_data(is_training_data, sequence_df_array)
    final_df = repair_values(is_training_data, complete_df)
    
    final_dataframe_deduplicated = final_df.drop_duplicates(subset=['ProxID', 'sequence_type', 'pos'], keep = 'first')
    save_data_to_directory(is_training_data, final_dataframe_deduplicated)
    
main()