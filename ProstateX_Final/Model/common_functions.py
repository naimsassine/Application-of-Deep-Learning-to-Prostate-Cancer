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

from ProstateX_Final.Model.grad_cam_code import make_gradcam_heatmap

from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')
from scipy.ndimage import morphology
import tensorflow
from ProstateX_Final.Model.deep_learning_model import *

# In this file, all the functions that were necessary for both models (detection and detection/classificatoin) are implemented : 
# from preparing the data, to loading it, to the metrics, to the loss functions ....
# Majority of loss functions inspired by : https://github.com/mirzaevinom/promise12_segmentation/blob/master/codes/metrics.py 

# functions for training : 

# taken from https://github.com/mirzaevinom/promise12_segmentation/blob/master/codes/metrics.py
def dice_coef(y_true, y_pred, smooth=1):
    # This loss function is known as the soft Dice loss because we directly use 
    # the predicted probabilities instead of thresholding and converting them into a binary mask.
    # explained at : https://www.jeremyjordan.me/semantic-segmentation/
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_true_f = tensorflow.cast(y_true_f, tensorflow.float32)
    y_pred_f = tensorflow.cast(y_pred_f, tensorflow.float32)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# taken from https://github.com/mirzaevinom/promise12_segmentation/blob/master/codes/metrics.py
def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


# inspired by https://github.com/mirzaevinom/promise12_segmentation/blob/master/codes/metrics.py
def dice_coef_dc(y_true, y_pred, smooth=1):
    # This loss function is known as the soft Dice loss because we directly use 
    # the predicted probabilities instead of thresholding and converting them into a binary mask.
    # explained at : https://www.jeremyjordan.me/semantic-segmentation/
    # The difference between this loss and the previous one is the following : when training the lesion detection and classification
    # model, the masks is divided in 3 channels, the first channel represents the background. I don't want to take into consideration this 
    # particular channel because in most of the images, the proportion of black pixels is huge compared to white pixels. So my score will
    # be high just because black pixels are predicted correctly, which will give less importance to the detected cancers. I don't want that
    # to happen
    y_true_f = K.flatten(y_true[:, :, :, 1:])
    y_pred_f = K.flatten(y_pred[:, :, :, 1:])
    y_true_f = tensorflow.cast(y_true_f, tensorflow.float32)
    y_pred_f = tensorflow.cast(y_pred_f, tensorflow.float32)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss_dc(y_true, y_pred):
    return 1-dice_coef_dc(y_true, y_pred)

# functions used for validation and testing : 


# the compute dsc is taken from https://www.programcreek.com/python/?CodeExample=compute+dice
def dice_metric(y_true, y_pred, smooth=1.0, per_element=False):
    # dsc, but here on numpys not on tensors
    # here we need to use the DSC and not soft dice, so we need to binarise
    # in order to binarise, we need to search for the threshold that will give us the best DSC
    # then, when we compute the DSC, we can use the intersection and union functions of numpy

    # I am directly adapting the dice metric to the masks that have 3 channels with the first as a backgorud
    # this function won't work if I am only testing lesion detection, and not detection and classification together
    # I will need to modify the fact that I don't want the first channel to make it work
    def compute_dsc(y_true, y_pred, smooth=1.0) : 
        volume_sum = y_true.sum() + y_pred.sum()
        if volume_sum == 0:
            return np.NaN
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
        volume_intersect = (y_true & y_pred).sum()
        return 2*volume_intersect / volume_sum 
    
    # let's determine that threshold
    list_of_thresholds = np.arange(0, 1, 0.01).tolist()
    list_of_max_dsc = []
    for i in list_of_thresholds :
        thresh = i
        maxval = 1
        y_pred_bin = (y_pred > thresh)
        if per_element == True : 
            list_of_max_dsc.append(compute_dsc(y_true[:, :, 1:], y_pred_bin[:, :, 1:]))
        else : 
            list_of_max_dsc.append(compute_dsc(y_true[:, :, :, 1:], y_pred_bin[:, :, :, 1:]))
    
    max_dsc = max(list_of_max_dsc)
    return max_dsc


# Inspired by : https://github.com/mirzaevinom/promise12_segmentation/blob/master/codes/metrics.py 
def rel_abs_vol_diff(y_true, y_pred):
    n_black_pixels = y_true.sum()
    if n_black_pixels == 0 : 
        val = np.abs( (y_pred.sum()/np.size(y_true) - 1)*100)
    else : 
        val = np.abs( (y_pred.sum()/y_true.sum() - 1)*100)
    return val

# intersection over union
# to really undertsand the difference between DSC and IOU, visit : 
# https://stats.stackexchange.com/questions/273537/f1-dice-score-vs-iou/276144#276144
# inpsired by https://github.com/bnsreenu/python_for_microscopists/blob/master/207_train_simple_unet_for_mitochondria_using_Jacard.py
def mean_iou(n_classes, y_true, y_pred):
    # here the inputs are volumes of different plane images
    from keras.metrics import MeanIoU
    IOU_keras = MeanIoU(num_classes=n_classes)  
    IOU_keras.update_state(y_true, y_pred)
    print("Mean IoU =", IOU_keras.result().numpy())

    values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
    print(values)
    if n_classes == 2 : 
        class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[1,0])
        class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[0,1])

        print("IoU for class1 is: ", class1_IoU)
        print("IoU for class2 is: ", class2_IoU)
    else : 
        class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[1,0]+ values[2,0])
        class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[0,1]+ values[2,1])
        class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[0,2]+ values[1,2])
        print("IoU for class1 is: ", class1_IoU)
        print("IoU for class2 is: ", class2_IoU)
        print("IoU for class3 is: ", class3_IoU)

# AUC and ROC can only be used for classifcation
# ROC is a probability curve and AUC represents the degree or measure of separability
# It tells how much the model is capable of distinguishing between classes
# https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5
def roc_auc(y_true, y_pred):
    # here the inputs are volumes of different plane images
    from sklearn.metrics import roc_curve, auc # roc curve tools
    yt = y_true.flatten()
    yp = y_pred.flatten()

    fpr, tpr, _ = roc_curve(yt,yp)


    roc_auc = auc(fpr,tpr)
    fig, ax = plt.subplots(1,1)
    ax.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    ax.legend(loc="lower right")

# The Hausdorff distance is the longest distance you can be forced to travel by an 
# adversary who chooses a point in one of the two sets, from where you then must travel to the other set. 
# In other words, it is the greatest of all the distances from a point in one set to the closest point in the other set.
# Computing the Hausdorff distance using the function below inspired
# by : https://github.com/mirzaevinom/promise12_segmentation/blob/master/codes/metrics.py 
def surface_dist(input1, input2, sampling=1, connectivity=1):
    input1 = np.squeeze(input1)
    input2 = np.squeeze(input2)

    input_1 = np.atleast_1d(input1.astype(np.bool))
    input_2 = np.atleast_1d(input2.astype(np.bool))


    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    S = input_1 - morphology.binary_erosion(input_1, conn)
    Sprime = input_2 - morphology.binary_erosion(input_2, conn)


    dta = morphology.distance_transform_edt(~S,sampling)
    dtb = morphology.distance_transform_edt(~Sprime,sampling)

    sds = np.concatenate([np.ravel(dta[Sprime!=0]), np.ravel(dtb[S!=0])])

    return sds

def number_of_detected_lesions(var, y_test, y_pred):
    # some statistics on the number of correctly detected lesions ...
    returned_list = []
    if var == "detection":
        n_correct_lesions = 0
        for i in range(len(y_test)) :
            contours = measure.find_contours(y_pred[i, :, :], 0.4) 
            if len(np.unique(y_test[i, :, :])) > 1 and len(contours) > 0 : 
                n_correct_lesions += 1

        returned_list.append(n_correct_lesions)
        return returned_list

    elif var == "detection_and_classification" : 
        n_correct_lesions = 0
        n_correct_lesions_sign = 0
        n_correct_lesions_non_sign = 0

        for i in range(len(y_test)) : 
            contours_none_sign = measure.find_contours(y_pred[i, :, :, 0], 0.4)
            contours_sign = measure.find_contours(y_pred[i, :, :, 1], 0.4) 
            if len(np.unique(y_test[i, :, :, 0])) > 1 and len(contours_none_sign) > 0 : 
                n_correct_lesions += 1
                n_correct_lesions_non_sign += 1
            if len(np.unique(y_test[i, :, :, 1])) > 1 and len(contours_sign) > 0  :
                n_correct_lesions += 1
                n_correct_lesions_sign += 1
        
        n_total_lesions = 0
        n_total_lesions_sign = 0
        n_total_lesions_non_sign = 0
        for i in range(len(y_test)) : 
            if len(np.unique(y_test[i, :, :, 0])) > 1 : 
                n_total_lesions += 1
                n_total_lesions_non_sign += 1
            if len(np.unique(y_test[i, :, :, 1])) > 1 : 
                n_total_lesions += 1
                n_total_lesions_sign += 1


        returned_list.append([n_correct_lesions, n_correct_lesions_non_sign, n_correct_lesions_sign])
        returned_list.append([n_total_lesions, n_total_lesions_non_sign, n_total_lesions_sign])
        return returned_list
    else : 
        return None

# Common functions : 
# Inspired by https://github.com/mirzaevinom/promise12_segmentation/blob/master/codes/metrics.py 
def check_predictions(data_set="test",y_pred="y_pred", var="type_of_test"):
    if not os.path.isdir('../images'):
        os.mkdir('../images')

    if data_set == "train" : 
        X = np.load('data/X_train.npy')
        y_true = np.load('data/y_train.npy')
    elif data_set == "test" : 
        X = np.load('data/X_test.npy')
        y_true = np.load('data/y_test.npy')
    else : 
        X = np.load('data/X_val.npy')
        y_true = np.load('data/y_val.npy')


    print('Results on' + data_set + 'set:')
    print('Accuracy:', dice_metric(y, y_pred))

    vol_scores = []
    ravd = []
    scores = []
    hauss_dist = []
    mean_surf_dist = []

    for i in range(len(y_pred)) :
        y_pred[i] = resize_pred_to_val(y_pred[i], y_true[i].shape)

        ravd.append( rel_abs_vol_diff( y_true[i] , y_pred[i] ) )
        vol_scores.append( dice_metric( y_true[i] , y_pred[i] , axis=None) )
        surfd = surface_dist(y_true[i] , y_pred[i], sampling=spacing)
        hauss_dist.append( surfd.max() )
        mean_surf_dist.append(surfd.mean())
        axis = tuple( range(1, y_true[i].ndim ) )
        scores.append( dice_metric( y_true[i], y_pred[i] , axis=axis) )

    ravd = np.array(ravd)
    vol_scores = np.array(vol_scores)
    scores = np.concatenate(scores, axis=0)
    scores_pred = number_of_detected_lesions(var, y_true, y_pred)

    print('Mean volumetric DSC:', vol_scores.mean() )
    print('Median volumetric DSC:', np.median(vol_scores) )
    print('Std volumetric DSC:', vol_scores.std() )
    print('Mean Hauss. Dist:', np.mean(hauss_dist) )
    print('Mean MSD:', np.mean(mean_surf_dist) )
    print('Mean Rel. Abs. Vol. Diff:', ravd.mean())
    print('Scores on the nmber of detected lesions :', scores_pred)

# Inspired by https://github.com/mirzaevinom/promise12_segmentation/blob/master/codes/metrics.py 
def resize_pred_to_val(y_pred, shape):
    row = shape[1]
    col =  shape[2]

    resized_pred = np.zeros(shape)
    for mm in range(len(y_pred)):
        resized_pred[mm,:,:] =  cv2.resize( y_pred[mm,:,:,0], (row, col), interpolation=cv2.INTER_NEAREST)

    return resized_pred.astype(int)

def average(lst):
  return sum(lst) / len(lst)

def d_b_centers(y_true, y_pred) : 
  # distance between centres
  distances = []
  for i in range(y_true.shape[0]):
    contours_true = measure.find_contours(y_true[i].reshape(128,128), 0.6)
    contours_pred = measure.find_contours(y_pred[i].reshape(128,128), 0.6)

    try : 
      lsqe_true = LsqEllipse()
      lsqe_true.fit(contours_true[0])

      lsqe_pred = LsqEllipse()
      lsqe_pred.fit(contours_pred[0])

      center_true, width, height, phi = lsqe_true.as_parameters()
      center_pred, width, height, phi = lsqe_pred.as_parameters()

      distance = math.sqrt( ((center_true[0]-center_pred[0])**2)+((center_true[1]-center_pred[1])**2) )
      distances.append(distance)
    except : 
      distances = distances

  return average(distances)

# prepare the data for training : detection and classification
def prepare_data_lesion_dc():
    all_images = []
    all_images = np.load('/Volumes/LaCie/Memoire/ProstateX/generated/mutli_class_segm/train_images_concatenated/train_images_concatenated.npy')

    masks = [] 
    masks = np.load('/Volumes/LaCie/Memoire/ProstateX/generated/mutli_class_segm/train_masks_concatenated/train_masks_concatenated.npy')

    train_masks = masks - 1
    train_masks = np.where(train_masks==-1, 0, train_masks)

    final_masks = np.zeros((train_masks.shape[0], 128, 128, 3))

    # one hot encoding done by hand : for each mask I need to create a mask of 3 channels, one for the background, one for 
    # significant lesions and one for none_significant lesions
    for k in range(len(train_masks)) : 
        for i in range(128) : 
            for j in range(128) : 
                if train_masks[k][i][j] == 1 :
                    final_masks[k][i][j][1] = 1
                elif train_masks[k][i][j] == 2 :
                    final_masks[k][i][j][2] = 1
                elif train_masks[k][i][j] == 0 : 
                    final_masks[k][i][j][0] = 1
                else :
                    print("Erronous Pixel value detected!!")

    n_imgs = len(masks)
    perc_test = int(n_imgs*15/100)

    import random
    list_of_test = []
    while len(list_of_test)<perc_test:
        n = random.randint(0,n_imgs-1)    
        if n not in list_of_test :
            list_of_test.append(n) 


    perc_val = int((n_imgs-perc_test)*10/100)

    list_of_val = []
    while len(list_of_val)<perc_val:
        n = random.randint(0,(n_imgs-1-perc_test))    
        if n not in list_of_val :
            list_of_val.append(n)

    list_of_train = list( set(range(n_imgs)) - set(list_of_test) - set(list_of_val))

    training_masks = final_masks[list_of_train]
    train_images = all_images[list_of_train]

    testing_masks = final_masks[list_of_test]
    test_images = all_images[list_of_test]

    validation_masks = final_masks[list_of_val]
    validation_images = all_images[list_of_val]

    mu_0 = np.mean(train_images[0]) 
    sigma_0 = np.std(train_images[0])

    mu_1 = np.mean(train_images[1]) 
    sigma_1 = np.std(train_images[1])

    mu_2 = np.mean(train_images[2]) 
    sigma_2 = np.std(train_images[2])

    mu_3 = np.mean(train_images[3]) 
    sigma_3 = np.std(train_images[3])

    train_images[0] = (train_images[0] - mu_0)/sigma_0
    train_images[1] = (train_images[1] - mu_1)/sigma_1
    train_images[2] = (train_images[2] - mu_2)/sigma_2
    train_images[3] = (train_images[3] - mu_3)/sigma_3

    open('normalization_values.txt', 'w').close()
    f = open("normalization_values.txt", "a")
    f.write(str(mu_0) + "\n")
    f.write(str(sigma_0) + "\n")

    f.write(str(mu_1) + "\n")
    f.write(str(sigma_1) + "\n")

    f.write(str(mu_2) + "\n")
    f.write(str(sigma_2) + "\n")

    f.write(str(mu_3) + "\n")
    f.write(str(sigma_3) + "\n")
    f.close()

    validation_images[0] = (validation_images[0] - mu_0)/sigma_0
    validation_images[1] = (validation_images[1] - mu_1)/sigma_1
    validation_images[2] = (validation_images[2] - mu_2)/sigma_2
    validation_images[3] = (validation_images[3] - mu_3)/sigma_3

    test_images[0] = (test_images[0] - mu_0)/sigma_0
    test_images[1] = (test_images[1] - mu_1)/sigma_1
    test_images[2] = (test_images[2] - mu_2)/sigma_2
    test_images[3] = (test_images[3] - mu_3)/sigma_3


    np.save('data/detection_and_classification/X_train.npy', train_images)
    np.save('data/detection_and_classification/y_train.npy', training_masks)

    np.save('data/detection_and_classification/X_val.npy', validation_images)
    np.save('data/detection_and_classification/y_val.npy', validation_masks)

    np.save('data/detection_and_classification/X_test.npy', test_images)
    np.save('data/detection_and_classification/y_test.npy', testing_masks)


# prepare the data for training : detection
def prepare_data_lesion_d():
    all_images = []
    all_images = np.load('E:/Memoire/ProstateX/generated/mutli_class_segm/train_images_concatenated/train_images_concatenated.npy')

    masks = [] 
    masks = np.load('E:/Memoire/ProstateX/generated/mutli_class_segm/train_masks_concatenated/train_masks_concatenated.npy')

    train_masks_beta = masks - 1
    train_masks_beta = np.where(train_masks_beta==-1, 0, train_masks_beta)
    train_masks_beta = np.where(train_masks_beta==2, 1, train_masks_beta)
    train_masks_beta = np.expand_dims(train_masks_beta, axis=3)

    train_masks = np.concatenate([train_masks_beta[20:49], train_masks_beta[90:999], train_masks_beta[1100:1149], train_masks_beta[1180:1278], train_masks_beta[1300:1449], train_masks_beta[1500:1599], train_masks_beta[1622:1884]], axis = 0)
    train_images = np.concatenate([all_images[20:49], all_images[90:999], all_images[1100:1149], all_images[1180:1278], all_images[1300:1449], all_images[1500:1599], all_images[1622:1884]], axis = 0)

    testing_masks = np.concatenate([train_masks_beta[0:19], train_masks_beta[50:89], train_masks_beta[1000:1099], train_masks_beta[1150:1179], train_masks_beta[1150:1179], train_masks_beta[1279:1299], train_masks_beta[1450:1499], train_masks_beta[1600:1621]], axis = 0)
    test_images = np.concatenate([all_images[0:19], all_images[50:89], all_images[1000:1099], all_images[1150:1179], all_images[1279:1299], all_images[1450:1499], all_images[1600:1621]], axis = 0)


    perc = int(train_images.shape[0]*90/100)

    train_i = train_images[:perc]
    validation_i = train_images[perc:train_images.shape[0]]

    train_m = train_masks[:perc]
    validation_m = train_masks[perc:train_images.shape[0]]

    mu = np.mean(train_i) 
    sigma = np.std(train_i)
    open('normalization_values.txt', 'w').close()
    f = open("normalization_values.txt", "a")
    f.write(str(mu) + "\n")
    f.write(str(sigma))
    f.close()
    train_i = (train_i - mu)/sigma
    validation_i = (validation_i - mu)/sigma
    test_images = (test_images - mu)/sigma

    np.save('data/detection/X_train.npy', train_i)
    np.save('data/detection/y_train.npy', train_m)

    np.save('data/detection/X_val.npy', validation_i)
    np.save('data/detection/y_val.npy', validation_m)

    np.save('data/detection/X_test.npy', test_images)
    np.save('data/detection/y_test.npy', testing_masks)

# prepare the data for training, but function not used here. I built it to test something but finally I won't have time to
# test everything there is to test
def prepare_data_maskadd():
    # not used, but if you want to use it : just add randomization in the data
    all_images = []
    all_images = np.load('/content/gdrive/MyDrive/mthesis/train_images_concatenated/train_images_concatenated.npy')

    masks = [] 
    masks = np.load('/content/gdrive/MyDrive/mthesis/train_masks_concatenated/train_masks_concatenated.npy')


    train_masks_beta = masks - 1
    train_masks_beta = np.where(train_masks_beta==-1, 0, train_masks_beta)


    # adding a fourth dimension, wich is the prostate masks
    pros_masks = np.where(masks==2, 1, masks)
    pros_masks = np.where(pros_masks==3, 1, pros_masks)
    pros_masks = np.expand_dims(pros_masks, axis=3)
    all_images = np.append(all_images, pros_masks, axis=-1)

    train_masks_beta = np.expand_dims(train_masks_beta, axis=3)

    train_masks = train_masks_beta[0:1087]
    testing_masks = train_masks_beta[1088:1278]

    train_images = all_images[0:1087]
    test_images = all_images[1088:1278]



    perc = int(train_images.shape[0]*90/100)

    train_i = train_images[:perc]
    validation_i = train_images[perc:train_images.shape[0]]

    train_m = train_masks[:perc]
    validation_m = train_masks[perc:train_images.shape[0]]

    mu = np.mean(train_i) 
    sigma = np.std(train_i)
    open('normalization_values.txt', 'w').close()
    f = open("normalization_values.txt", "a")
    f.write(str(mu) + "\n")
    f.write(str(sigma))
    f.close()
    train_i = (train_i - mu)/sigma
    validation_i = (validation_i - mu)/sigma
    test_images = (test_images - mu)/sigma

    np.save('/content/gdrive/MyDrive/Colab Notebooks/SegmTrial/data/X_train.npy', train_i)
    np.save('/content/gdrive/MyDrive/Colab Notebooks/SegmTrial/data/y_train.npy', train_m)

    np.save('/content/gdrive/MyDrive/Colab Notebooks/SegmTrial/data/X_val.npy', validation_i)
    np.save('/content/gdrive/MyDrive/Colab Notebooks/SegmTrial/data/y_val.npy', validation_m)

    np.save('/content/gdrive/MyDrive/Colab Notebooks/SegmTrial/data/X_test.npy', test_images)
    np.save('/content/gdrive/MyDrive/Colab Notebooks/SegmTrial/data/y_test.npy', testing_masks)

def load_only_training(test_type):
    if test_type == "detection": 
        X_train = np.load('data/detection/X_train.npy')
        y_train = np.load('data/detection/y_train.npy')
        X_val = np.load('data/detection/X_val.npy')
        y_val = np.load('data/detection/y_val.npy')
    else :
        X_train = np.load('data/detection_and_classification/X_train.npy')
        y_train = np.load('data/detection_and_classification/y_train.npy')
        X_val = np.load('data/detection_and_classification/X_val.npy')
        y_val = np.load('data/detection_and_classification/y_val.npy')


    return X_train, y_train, X_val, y_val

def load_data(test_type):

    if test_type == "detection": 
        X_train = np.load('data/detection/X_train.npy')
        y_train = np.load('data/detection/y_train.npy')
        X_val = np.load('data/detection/X_val.npy')
        y_val = np.load('data/detection/y_val.npy')
        X_test = np.load('data/detection/X_test.npy')
        y_test = np.load('data/detection/y_test.npy')
    else :
        X_train = np.load('data/detection_and_classification/X_train.npy')
        y_train = np.load('data/detection_and_classification/y_train.npy')
        X_val = np.load('data/detection_and_classification/X_val.npy')
        y_val = np.load('data/detection_and_classification/y_val.npy')
        X_test = np.load('data/detection_and_classification/X_test.npy')
        y_test = np.load('data/detection_and_classification/y_test.npy')


    return X_train, y_train, X_val, y_val, X_test, y_test

# Inspired by https://github.com/mirzaevinom/promise12_segmentation
def augment_validation_data(X, y, seed=3):
    # this function won't probably be used
    img_rows = X.shape[1]
    img_cols =  X.shape[2]

    elastic = partial(elastic_transform, alpha=img_rows*1.5, sigma=img_rows*0.07 )
    data_gen_args = dict(preprocessing_function=elastic)

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    image_datagen.fit(X, seed=seed)
    mask_datagen.fit(y, seed=seed)

    image_generator = image_datagen.flow(X, batch_size=100, seed=seed)
    mask_generator = mask_datagen.flow(y, batch_size=100, seed=seed)

    train_generator = zip(image_generator, mask_generator)

    count=0
    X_val = []
    y_val = []

    for X_batch, y_batch in train_generator:

        if count==5:
            break

        count+=1

        X_val.append(X_batch)
        y_val.append(y_batch)

    X_val = np.concatenate(X_val, axis=0)
    y_val = np.concatenate(y_val, axis=0)
    return X_val, y_val

def get_model_dc(img_rows, img_cols, img_depth, path):
    model = UNet((img_rows, img_cols, img_depth), start_ch=8, out_ch=3, depth=4, batchnorm=True, dropout=0.5, maxpool=True, residual=True)
    model.load_weights(path)
    model.compile(  optimizer=Adam(), loss=dice_coef_loss, metrics=[dice_coef])
    return model

def get_model_d(img_rows, img_cols, img_depth, path):
    model = UNet((img_rows, img_cols, img_depth), start_ch=8, out_ch=1, depth=4, batchnorm=True, dropout=0.5, maxpool=True, residual=True)
    model.load_weights(path)
    model.compile(  optimizer=Adam(), loss=dice_coef_loss, metrics=[dice_coef])
    return model

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



import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from skimage import measure

def compute_scores(y_true, y_pred):
    coef = []
    for i in range(len(y_true)):
        coef.append(dice_metric(y_true[i], y_pred[i], per_element=True))
    return coef

# Inspired by https://github.com/mirzaevinom/promise12_segmentation
def make_plots_none_sign(X, y, y_pred, n_best=30, n_worst=20):
    img_rows = X.shape[1]
    img_cols = img_rows
    scores = compute_scores(y, y_pred)
    axis =  tuple( range(1, X.ndim ) )
    sort_ind = np.argsort( scores )[::-1]
    indice = np.nonzero( y.sum(axis=axis) )[0]
    img_list = []
    count = 1
    for ind in sort_ind:
        if ind in indice:
            img_list.append(ind)
            count+=1
        if count>n_best:
            break

    segm_pred = y_pred[img_list, :, :, 1]
    img = X[img_list][:, :, :, 0]
    segm = y[img_list, :, :, 1].astype('float32')

    n_cols= 4
    n_rows = int( np.ceil(len(img)/n_cols) )
    print(img_list)
    fig = plt.figure(figsize=[ 4*n_cols, int(4*n_rows)] )
    gs = gridspec.GridSpec( n_rows , n_cols )
    for mm in range( len(img) ):

        ax = fig.add_subplot(gs[mm])
        ax.imshow(img[mm] )
        contours = find_contours(segm[mm, :, :], 0.01, fully_connected='high')

        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='r')

        contours = find_contours(segm_pred[mm, :, :], 0.01, fully_connected='high')
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='b')


        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1) 

# Inspired by https://github.com/mirzaevinom/promise12_segmentation
def make_plots_sign(X, y, y_pred, n_best=30, n_worst=20):
    img_rows = X.shape[1]
    img_cols = img_rows
    scores = compute_scores(y, y_pred)
    axis =  tuple( range(1, X.ndim ) )
    sort_ind = np.argsort( scores )[::-1]
    indice = np.nonzero( y.sum(axis=axis) )[0]
    img_list = []
    count = 1
    for ind in sort_ind:
        if ind in indice:
            img_list.append(ind)
            count+=1
        if count>n_best:
            break


    segm_pred = y_pred[img_list, :, :, 2]
    img = X[img_list][:, :, :, 0]
    segm = y[img_list, :, :, 2].astype('float32')

    n_cols= 4
    n_rows = int( np.ceil(len(img)/n_cols) )

    fig = plt.figure(figsize=[ 4*n_cols, int(4*n_rows)] )
    gs = gridspec.GridSpec( n_rows , n_cols )
    for mm in range( len(img) ):

        ax = fig.add_subplot(gs[mm])
        ax.imshow(img[mm] )
        contours = find_contours(segm[mm, :, :], 0.01, fully_connected='high')

        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='violet')

        contours = find_contours(segm_pred[mm, :, :], 0.01, fully_connected='high')
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='orange')


        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1) 

