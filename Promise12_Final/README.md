As previously explained, in this folder you will find everything related to the model that achieves prostate segmentation. Here is a 
brief explanation of what each file contains

- deep_learning_model.py : contains the architecture of the model, the U-Net Model that is built for this task as long as a detailed explanation on what each parameter represents and why it was chosen
- exp_and_results.py : contains all the functions that were necessary to implement for training, testing and evaluating the model. The different metrics as well as training function are implemented here. Also, the functions that are meant to run the tests can be found in this file
- pre_processing_data.py : from the downloaded dataset, to a usable numpy. This file contains all the functions that need to be applied in order to pre-process the integrity of the downloaded images and data
- training_mode.py : everything involving training the different models
