# The goal of the functions below is to take the MRIs provided by the
# Promise12 challenge and pre-process them to finally save the training/testing
# data into numpy arrays
import os
import cv2
import numpy as np
import SimpleITK as sitk 
import random

# The method iteslf was inspired by : https://github.com/mirzaevinom/promise12_segmentation/blob/master/codes/metrics.py 
# But I re-built most of the codes myself because my logic didn't fully align with the source's writer's logic for this file. 
# So the code is heavily inspired by the above source, but I re-built, re-structured and re-wrote most of the functions in this file myself


def elastic_transform(image, x=None, y=None, alpha=256*3, sigma=256*0.07):
    # inpired by https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a#file-elastic_transform-py and https://github.com/mirzaevinom/promise12_segmentation/blob/master/codes/metrics.py 
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """

    shape = image.shape
    blur_size = int(4*sigma) | 1
    dx = cv2.GaussianBlur((np.random.rand(shape[0],shape[1]) * 2 - 1), ksize=(blur_size, blur_size), sigmaX=sigma)* alpha
    dy = cv2.GaussianBlur((np.random.rand(shape[0],shape[1]) * 2 - 1), ksize=(blur_size, blur_size), sigmaX=sigma)* alpha

    if (x is None) or (y is None):
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')

    map_x =  (x+dx).astype('float32')
    map_y =  (y+dy).astype('float32')

    return cv2.remap(image.astype('float32'), map_y,  map_x, interpolation=cv2.INTER_NEAREST).reshape(shape)

def resize_image(imgs, img_rows, img_cols):
    # We resize the MRI slices to a dimension of 256x256 to have all images at a same shape
    new_imgs = np.zeros([len(imgs), img_rows, img_cols])
    for mm, img in enumerate(imgs):
        new_imgs[mm] = cv2.resize( img, (img_rows, img_cols), interpolation=cv2.INTER_NEAREST)
    return new_imgs


def generate_val_list():
    randomlist = []
    for i in range(0,5):
        # 41 cases because 8 cases are left for testing
        n = random.randint(0,41)
        randomlist.append(n)
    return randomlist


def save_to_array(img_list, type_of_set):
    fileList =  os.listdir('E:/Memoire/Segmentation/promise12_segmentation-master/MasterThesis/data/train/') 
    fileList = list(filter(lambda x: '.mhd' in x, fileList))
    fileList.sort()

    images = []
    masks = []

    filtered = filter(lambda x: any(str(ff).zfill(2) in x for ff in img_list), fileList)

    for filename in filtered:
        if filename[0] != "." :
            itkimage = sitk.ReadImage('E:/Memoire/Segmentation/promise12_segmentation-master/MasterThesis/data/train/'+filename)
            imgs = sitk.GetArrayFromImage(itkimage)

            if 'segm' in filename.lower():
                imgs= resize_image(imgs, 256, 256)
                masks.append( imgs )

            else:
                imgs = resize_image(imgs, 256, 256)
                images.append(imgs)

    images = np.concatenate( images , axis=0 ).reshape(-1, 256, 256, 1)
    masks = np.concatenate(masks, axis=0).reshape(-1, 256, 256, 1)
    masks = masks.astype(int)

    images = curv_denoising(images)
    if type_of_set == "train" : 
        mu = np.mean(images) 
        sigma = np.std(images)
        open('normalization_values.txt', 'w').close()
        f = open("normalization_values.txt", "a")
        f.write(str(mu) + "\n")
        f.write(str(sigma))
        f.close()
    else : 
        f = open("normalization_values.txt", "r")
        mu = float(f.readline())
        sigma = float(f.readline())
        f.close()

    images = (images - mu)/sigma

    if type_of_set == "train" : 
        np.save('data/X_train.npy', images)
        np.save('data/y_train.npy', masks)
    elif type_of_set == "test" : 
        np.save('data/X_test.npy', images)
        np.save('data/y_test.npy', masks)
    else : 
        np.save('data/X_val.npy', images)
        np.save('data/y_val.npy', masks)


def curv_denoising(imgs):
    # inspired by https://github.com/mirzaevinom/promise12_segmentation/blob/master/codes/augmenters.py#L31
    # define the parametres
    timeStep=0.186
    numberOfIterations=5

    for mm in range(len(imgs)):
        img = sitk.GetImageFromArray(imgs[mm])
        img = sitk.CurvatureFlow(image1=img,
                                        timeStep=timeStep,
                                        numberOfIterations=numberOfIterations)

        imgs[mm] = sitk.GetArrayFromImage(img)


    return imgs


def convert_img_to_array():
    testing_list = [42, 43, 44, 45, 46, 47, 48, 49]
    val_list = generate_val_list()
    train_list = list( set(range(42)) - set(val_list))

    save_to_array(train_list, "train")
    save_to_array(testing_list, "test")
    save_to_array(val_list, "val")


def load_data():

    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    X_val = np.load('data/X_val.npy')
    y_val = np.load('data/y_val.npy')
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')

    return X_train, y_train, X_val, y_val, X_test, y_test

# I load only training sometimes, when I don't want to test for expample, so 
# I gain time since I don't import the testing set also
def load_only_training():
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    X_val = np.load('data/X_val.npy')
    y_val = np.load('data/y_val.npy')

    return X_train, y_train, X_val, y_val

