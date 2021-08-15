# This file contains the different functions that were used to analyse the results that were obtained after training
from __future__ import division, print_function

from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')
import numpy as np
from scipy.ndimage import morphology
import tensorflow
from tensorflow.keras.optimizers import Adam


from Promise12_Final.pre_processing_data import *
from Promise12_Final.deep_learning_model import *

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


# inspired by https://github.com/mirzaevinom/promise12_segmentation/blob/master/codes/metrics.py 
def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


# functions to see the results of the testing

# the compute dsc is taken from https://www.programcreek.com/python/?CodeExample=compute+dice
def dice_metric(y_true, y_pred, smooth=1.0):
    # dsc, but here on numpys not on tensors
    # here we need to use the DSC and not soft dice, so we need to binarise
    # in order to binarise, we need to search for the threshold that will give us the best DSC
    # then, when we compute the DSC, we can use the intersection and union functions of numpy

    def compute_dsc(y_true, y_pred, smooth=1.0) : 
        volume_sum = y_true.sum() + y_pred.sum()
        if volume_sum == 0:
            return np.NaN
        volume_intersect = (y_true & y_pred).sum()
        return 2*volume_intersect / volume_sum 
    
    # let's determine that threshold
    list_of_thresholds = np.arange(0, 1, 0.01).tolist()
    list_of_max_dsc = []
    for i in list_of_thresholds :
        thresh = i
        maxval = 1
        y_pred_bin = (y_pred > thresh) * maxval
        list_of_max_dsc.append(compute_dsc(y_true, y_pred_bin))
    
    max_dsc = max(list_of_max_dsc)
    print("this is the threshold that gives the maximum dsc : ", list_of_thresholds[list_of_max_dsc.index(max_dsc)])
    return max_dsc



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
    # to adjust corresponding to the number of classes
    print(values)
    class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[1,0])
    class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[0,1])

    print("IoU for class1 is: ", class1_IoU)
    print("IoU for class2 is: ", class2_IoU)

# inspired by : https://github.com/mirzaevinom/promise12_segmentation/blob/master/codes/metrics.py 
def rel_abs_vol_diff(y_true, y_pred):
    return np.abs( (y_pred.sum()/y_true.sum() - 1)*100)

# The Hausdorff distance is the longest distance you can be forced to travel by an 
# adversary who chooses a point in one of the two sets, from where you then must travel to the other set. 
# In other words, it is the greatest of all the distances from a point in one set to the closest point in the other set. (wikipedia)
# The way I compute the haussdorff distance here is uing the following function, taken from https://github.com/mirzaevinom/promise12_segmentation/blob/master/codes/metrics.py 
def surface_dist(input1, input2, sampling=1, connectivity=1):
    input1 = np.squeeze(input1) # Remove axes of length one from the given input
    input2 = np.squeeze(input2)

    input_1 = np.atleast_1d(input1.astype(np.bool)) 
    # Convert inputs to arrays with at least one dimension.
    # Scalar inputs are converted to 1-dimensional arrays, whilst higher-dimensional inputs are preserved.
    input_2 = np.atleast_1d(input2.astype(np.bool))


    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)
    # Generate a binary structure for binary morphological operations.

    S = input_1 - morphology.binary_erosion(input_1, conn)
    Sprime = input_2 - morphology.binary_erosion(input_2, conn)


    dta = morphology.distance_transform_edt(~S,sampling) # distance to a
    dtb = morphology.distance_transform_edt(~Sprime,sampling) # distance to b
    # ravel : Returns a contiguous flattened array.
    # ouput distance to "a" where we have a "b", and then distance to "b" when we have an "a"
    sds = np.concatenate([np.ravel(dta[Sprime!=0]), np.ravel(dtb[S!=0])])

    return sds


# functions used for testing

# inspired by : https://github.com/mirzaevinom/promise12_segmentation/blob/master/codes/metrics.py 
def check_predictions(the_list, data_set="test", y_pred=[]):

    if not os.path.isdir('../images'):
        os.mkdir('../images')

    if data_set == "train" : 
        X = np.load('data/X_train.npy')
        y = np.load('data/y_train.npy')
    elif data_set == "test" : 
        X = np.load('data/X_test.npy')
        y = np.load('data/y_test.npy')
    else : 
        X = np.load('data/X_val.npy')
        y = np.load('data/y_val.npy')


    print('Results on ' + data_set + ' set:')
    print('Accuracy:', dice_metric(y, y_pred))


    vol_scores = []
    ravd = []
    scores = []
    hauss_dist = []
    mean_surf_dist = []

    start_ind = 0
    end_ind  = 0
    for y_true, spacing in read_cases(the_list):

        start_ind = end_ind
        end_ind +=len(y_true)

        y_pred_up = resize_pred_to_val( y_pred[start_ind:end_ind], y_true.shape)

        ravd.append( rel_abs_vol_diff( y_true , y_pred_up ) )
        vol_scores.append( dice_metric( y_true , y_pred_up , axis=None) )
        surfd = surface_dist(y_true , y_pred_up, sampling=1)
        hauss_dist.append( surfd.max() )
        mean_surf_dist.append(surfd.mean())
        axis = tuple( range(1, y_true.ndim ) )
        scores.append( dice_metric( y_true, y_pred_up , axis=axis) )

    ravd = np.array(ravd)
    vol_scores = np.array(vol_scores)
    scores = np.concatenate(scores, axis=0)

    print('Mean volumetric DSC:', vol_scores.mean() )
    print('Median volumetric DSC:', np.median(vol_scores) )
    print('Std volumetric DSC:', vol_scores.std() )
    print('Mean Hauss. Dist:', np.mean(hauss_dist) )
    print('Mean MSD:', np.mean(mean_surf_dist) )
    print('Mean Rel. Abs. Vol. Diff:', ravd.mean())

# inspired by : https://github.com/mirzaevinom/promise12_segmentation/blob/master/codes/metrics.py 
def resize_pred_to_val(y_pred, shape):
    row = shape[1]
    col =  shape[2]

    resized_pred = np.zeros(shape)
    for mm in range(len(y_pred)):
        resized_pred[mm,:,:] =  cv2.resize( y_pred[mm,:,:,0], (row, col), interpolation=cv2.INTER_NEAREST)

    return resized_pred.astype(int)

# inspired by : https://github.com/mirzaevinom/promise12_segmentation/blob/master/codes/metrics.py 
def read_cases(the_list=None, folder='E:/Memoire/Segmentation/promise12_segmentation-master/MasterThesis/data/train/', masks=True):
    fileList =  os.listdir(folder)
    fileList = filter(lambda x: '.mhd' in x, fileList)
    if masks:
        fileList = filter(lambda x: 'segm' in x.lower(), fileList)
    fileList.sort()
    if the_list is not None:
        fileList = filter(lambda x: any(str(ff).zfill(2) in x for ff in the_list), fileList)

    for filename in fileList:
        itkimage = sitk.ReadImage(folder+filename)
        imgs = sitk.GetArrayFromImage(itkimage)
        yield imgs, itkimage.GetSpacing()[::-1]

# inspired by : https://github.com/mirzaevinom/promise12_segmentation/blob/master/codes/metrics.py 
def get_model(img_rows, img_cols, img_depth, path):
    model = UNet((img_rows, img_cols, img_depth), start_ch=8, depth=5, batchnorm=True, dropout=0.5, maxpool=True, residual=True)
    model.load_weights(path)
    model.compile(optimizer=Adam(), loss=dice_coef_loss, metrics=[dice_coef])
    return model


# functions to call in the main() function


def test_without_augmentation():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    model = get_model(256, 256, 1, 'data/weights_no_augmentation.h5')
    y_pred = model.predict(X_test, verbose=1, batch_size=128)
    check_predictions(the_list=[42, 43, 44, 45, 46, 47, 48, 49], data_set="test", y_pred=y_pred)

def test_with_augmentation():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    model = get_model(256, 256, 1, 'data/weights_with_augmentation.h5')
    y_pred = model.predict(X_test, verbose=1, batch_size=128)
    check_predictions(the_list=[42, 43, 44, 45, 46, 47, 48, 49], data_set="test", y_pred=y_pred)

# Inspired by https://github.com/mirzaevinom/promise12_segmentation
def numpy_dice(y_true, y_pred, axis=None, smooth=1.0):

    intersection = y_true*y_pred

    return ( 2. * intersection.sum(axis=axis) +smooth)/ (np.sum(y_true, axis=axis) + np.sum(y_pred, axis=axis) +smooth )

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from skimage import measure

# Inspired by https://github.com/mirzaevinom/promise12_segmentation
def make_plots(X, y, y_pred, n_best=30, n_worst=20):
    img_rows = X.shape[1]
    img_cols = img_rows
    axis =  tuple( range(1, X.ndim ) )
    scores = numpy_dice(y, y_pred, axis=axis )
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
    segm_pred = y_pred[img_list].reshape(-1,img_rows, img_cols)
    img = X[img_list][:, :, :, 0]
    segm = y[img_list].reshape(-1, img_rows, img_cols).astype('float32')

    n_cols= 4
    n_rows = int( np.ceil(len(img)/n_cols) )

    fig = plt.figure(figsize=[ 4*n_cols, int(4*n_rows)] )
    gs = gridspec.GridSpec( n_rows , n_cols )

    for mm in range( len(img) ):

        ax = fig.add_subplot(gs[mm])
        ax.imshow(img[mm] )
        contours = measure.find_contours(segm[mm], 0.01, fully_connected='high')
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='r')

        contours = measure.find_contours(segm_pred[mm], 0.01, fully_connected='high')
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='b')

        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1)  
