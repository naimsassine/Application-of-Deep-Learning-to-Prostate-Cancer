Welcome to the repository of my master thesis : the application of Deep Learning to prostate cancer

Prostate carcinoma or prostate cancer is one of the most prevalent cancers worldwide. Invasive techniques are usually used to diagnose prostate cancer. Non-invasive methods have the advantage of causing very little damage to the patients and producing faster results. Multi-parametric magnetic resonance imaging (mp-MRI) is a non-invasive technique used for detecting, classifying, and quantifying prostate lesions. The produced images are generally studied by radiologists in order to diagnose prostate cancer. This diagnosis suffers from erroneous detection or interpretation due to factors like the observer's limitations, the image quality, the complexity of the clinical cases and variability in lesions appearance. Deep Learning (DL), a subset of Artificial Intelligence is a technology that is currently used to uncover and study the patterns behind features of images. DL has a potential of providing better standardization and consistency in identifying prostate lesions and improving prostate cancer diagnosis by studying mp-MRI scans. In this research, a computer aided software based on deep learning is built, trained and tested to produce insights that could potentially help doctors improve their diagnoses of prostate cancer. The three main tasks that this software achieves are prostate segmentation, lesion detection and lesion classification into significant or non-significant lesions. Two DL networks were conceived to accomplish these three tasks, each achieving a Dice Similarity Coefficient (DSC) of 87% and 61% respectively. These results show that this built software based on two DL models provides accurate predictions and insights that could be used by doctors to improve their diagnosis of prostate cancer.

Here is a quick description of the structure of the repository : 

1- Promise12-Final : in this folder you can find all the codes that were used to solve the prostate segmentation task

2- ProstateX-Final : in this folder you can find all the codes that were used to solve the lesion detection/classification task

3- Traces : this folder contains all the codes that I built for testing some features or some methods, but that did not make it into the final code becaues they were not good ennough, or because they were just meant to be used only for testing some other functionalities. It is important to note that many methods were used before reaching the final U-Net architecture and final preprocessing steps, and its in this folder "Traces" that you can find all the different struggles that I went through to finally get the final version of my code

4- The Pipleline.py is the code that connects both models

You can also find the written thesis as a pdf file above!
