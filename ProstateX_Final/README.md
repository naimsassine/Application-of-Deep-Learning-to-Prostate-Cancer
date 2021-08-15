The PreProcessing folder contains all the functions that were used to preprocess the downloaded data like described in the thesis
The Prostate_X_Preprocessing contains python codes that I used from an open source repository on github that helped me pre-process the 
downloaded data of the ProstateX challenge. In fact the different methods used help convert the images to the right format
in order to be used later on in the Final_PreProcessing folder for the finalisation steps.

Here is the repository that helped built these functions : https://github.com/alexhamiltonRN. Not that this repository's goal is to solve the ProstateX challenge, but I did not follow at all what they did, I just took inspiration of how they pre-processed the images.

In the model folder, you can find all the functions that were built and used to train and evaluate the different models, for lesion detection and detection/classification



In this folder, everything related to the detection and classificatoin of lesions is aborded
Please keep in mind that both tasks are done in a single run, with a model that achieves semantic segmentation

In the PreProcessing folder, to subfolders exist : 
- The first is Prostate_X_Preprocessing : it contains a set of functions that I found within an open source repository on github : https://github.com/alexhamiltonRN (this repo may have been deleted, I consulted it in March 2021). Note that this repository contains a lot more than just pre-processing the data, but the technique developed was really different from the technique I inteded to develop within my master thesis, that is why I only inspired myself by the preprocessing functions found in this repo. These functions are essential since they allow to convert the downladed image of the ProstateX challenge into a format that is easily manipulable
- The second is Final_PreProcessing and only contains a file with a set of functions that allowed me to go from a set of mp-MRIS to trainable and testable MRIs and Masks. These are functions that I built myself and have multiple functions that are explained in the thesis, as well as detailed in the code

In the Model folder, everything else is present. From the model, to the training functions, to the data augmentation, to the metrics and finally the testing functinos : 
- common_functions : all the functions that are commonly used for training, testing, validating, drawing plots ...
- deep_learing_model : the U-Net model
- grad_cam_code : the code for the grad-cam implementation
- LesionDetection and LesionDetectionAndClassification : as explained in the first version of my master thesis (not the current one) I first tested my data as well as my model on only localising the lesions, and not differentiating between significant and none-significant. I did that to see of the model would first of all learn to segment a general lesion and not a specific one. Then once that was done, I tested the full semantic segmentation of significant and non-significant lesions. That is why there exists two files and sometimes two functions : one for only lesion detection, and the other for lesion detection and classification. In these functions, the models are trained and the test functions are built