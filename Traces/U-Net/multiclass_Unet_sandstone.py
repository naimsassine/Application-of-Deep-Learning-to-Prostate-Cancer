from simple_multi_unet_model import multi_unet_model #Uses softmax 

from keras.utils import normalize
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pdb
import keras
import tensorflow
from keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
from metrics import dice_coef, dice_coef_loss
import math
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import np_utils

def weightedLoss(originalLossFunc, weightsList):

    def lossFunc(true, pred):
        axis = -1 #if channels last 
        #axis=  1 #if channels first


        #argmax returns the index of the element with the greatest value
        #done in the class axis, it returns the class index    
        classSelectors = K.argmax(true, axis=axis) 
            #if your loss is sparse, use only true as classSelectors


        #considering weights are ordered by class, for each class
        #true(1) if the class index is equal to the weight index 
        # here doesn't work  
        one64 = np.ones(1, dtype=np.int64)
        classSelectors = [K.equal(one64[0]*i, classSelectors) for i in range(len(weightsList))]
        #classSelectors = [K.equal(i, classSelectors) for i in range(len(weightsList))]
        # modify the files, if it doesn't work try anaconda

        #casting boolean to float for calculations  
        #each tensor in the list contains 1 where ground true class is equal to its index 
        #if you sum all these, you will get a tensor full of ones. 
        classSelectors = [K.cast(x, K.floatx()) for x in classSelectors]

        #for each of the selections above, multiply their respective weight
        weights = [sel * w for sel,w in zip(classSelectors, weightsList)] 

        #sums all the selections
        #result is a tensor with the respective weight for each element in predictions
        weightMultiplier = weights[0]
        for i in range(1, len(weights)):
            weightMultiplier = weightMultiplier + weights[i]


        #make sure your originalLossFunc only collapses the class axis
        #you need the other axes intact to multiply the weights tensor
        loss = originalLossFunc(true,pred) 
        loss = loss * weightMultiplier

        return loss
    return lossFunc


def compute_weights(counts, nb_classes):
    div = 128*128
    weights = np.array([])
    if len(counts) == 1 : # only 0 value pixels
        weights = np.append(weights, counts[0])
        weights = np.append(weights, 0)
        weights = np.append(weights, 0)
        weights = np.append(weights, 0)
    elif len(counts) == 2 : # 0 and 1 value pixels
        weights = np.append(weights, counts[0])
        weights = np.append(weights, counts[1])
        weights = np.append(weights, 0)
        weights = np.append(weights, 0)
    elif len(counts) == 3 : # 0, 1 and 2 value pixels
        weights = np.append(weights, counts[0])
        weights = np.append(weights, counts[1])
        weights = np.append(weights, counts[2])
        weights = np.append(weights, 0)
    elif len(counts) == 4 : # 0, 1 , 2 and 3 value pixels
        weights = np.append(weights, counts[0])
        weights = np.append(weights, counts[1])
        weights = np.append(weights, counts[2])
        weights = np.append(weights, counts[3])
    return np.divide(weights, div)[:nb_classes]


#Resizing images, if needed
SIZE_X = 128 
SIZE_Y = 128
n_classes=2 #Number of classes for segmentation


#Capture training image info as a list
train_images = []

#Convert list to array for machine learning processing        
train_images = np.load('E:/Memoire/ProstateX/generated/mutli_class_segm/train_images_concatenated/train_images_concatenated.npy')

#Capture mask/label info as a list
train_masks = [] 

#Convert list to array for machine learning processing          
train_masks = np.load('E:/Memoire/ProstateX/generated/mutli_class_segm/train_masks_concatenated/train_masks_concatenated.npy')


#train_masks = np.where(train_masks==3, 1, train_masks)
#train_masks = np.where(train_masks==2, 1, train_masks)
train_masks = train_masks - 1
train_masks = np.where(train_masks==-1, 0, train_masks)
train_masks = np.where(train_masks==2, 1, train_masks)
###############################################
#Encode labels... but multi dim array so need to flatten, encode and reshape
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
n, h, w = train_masks.shape
train_masks_reshaped = train_masks.reshape(-1,1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

np.unique(train_masks_encoded_original_shape)

#################################################
train_images = np.expand_dims(train_images, axis=4)
train_images = normalize(train_images, axis=1)

train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)

#Create a subset of data for quick testing
#Picking 10% for testing and remaining for training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_images, train_masks_input, test_size = 0.20, random_state = 0)


print("Class values in the dataset are ... ", np.unique(y_train))  # 0 is the background/few unlabeled 

from keras.utils import to_categorical
train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))



test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))



from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',np.unique(train_masks_reshaped_encoded),train_masks_reshaped_encoded)
print("Class weights are...:", class_weights)
#class_weights = {i : class_weights[i] for i in range(n_classes)}

def add_sample_weights(image, label, class_weights):
  # The weights for each class, with the constraint that:
  #     sum(class_weights) == 1.0
  class_weights = tensorflow.constant(class_weights)
  class_weights = class_weights/tensorflow.reduce_sum(class_weights)

  # Create an image of `sample_weights` by using the label at each pixel as an 
  # index into the `class weights` .
  sample_weights = tensorflow.gather(class_weights, indices=tensorflow.cast(label, tensorflow.int32))

  return sample_weights

y_train_reshape = y_train.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2]))
X_train_reshape = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3]))
sample_weights = add_sample_weights(X_train_reshape, y_train_reshape, class_weights)


# compute sample weights
"""
sample_weights = []
pdb.set_trace()
for i in range(y_train.shape[0]):
    unique, counts = np.unique(y_train[i], return_counts=True)
    computed_weights = compute_weights(counts, n_classes)
    sample_weights = sample_weights + [computed_weights]

sample_weights = np.stack(sample_weights)
"""

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]




def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

model = get_model()
#model.compile(optimizer='adam', loss= weightedLoss(keras.losses.categorical_crossentropy, class_weights), metrics=['accuracy'])
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='adam', loss=tensorflow.keras.losses.MeanSquaredError(), metrics=['accuracy'])
#model.compile(optimizer=Adam(lr=0.001), loss=weightedLoss(dice_coef_loss, class_weights), metrics=[dice_coef])
opt = SGD(lr=0.005)
model.compile(optimizer=opt, loss=tensorflow.keras.losses.MeanSquaredError(), metrics=['accuracy'])
#model.compile(optimizer=opt, loss='mse', metrics=[tensorflow.keras.metrics.MeanIoU(num_classes=2)])
model.summary()
# the problem with categorical cross entropy 


#If starting with pre-trained weights. 
#model.load_weights('???.hdf5')

#history = model.fit(X_train, y_train_cat, batch_size = 10, verbose=1, epochs=5, validation_data=(X_test, y_test_cat), shuffle=False)

# K-folds

Y = np.arange(0,y_train_cat.shape[0] , 1)          

kf = KFold(n_splits = 5, shuffle=True)


def get_model_name(k):
    return 'model_'+str(k)+'.h5'


VALIDATION_ACCURACY = []
VALIDATION_LOSS = []

save_dir = ''
fold_var = 1
for train_index, val_index in kf.split(Y):
    # CREATE CALLBACKS
    checkpoint = tensorflow.keras.callbacks.ModelCheckpoint(save_dir+get_model_name(fold_var), 
                            monitor='val_accuracy', verbose=1, 
                            save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    # There can be other callbacks, but just showing one because it involves the model name
    # This saves the best model
    # FIT THE MODEL
    x = train_index.shape[0]
    history = model.fit(X_train[train_index], y_train_cat[train_index],
                batch_size=10,
                epochs=5,
                sample_weight=sample_weights[:x],
                callbacks=callbacks_list,
                validation_data=(X_train[val_index], y_train_cat[val_index]))
    #PLOT HISTORY
    #		:
    #		:

    # LOAD BEST MODEL to evaluate the performance of the model
    model.load_weights("model_"+str(fold_var)+".h5")

    results = model.evaluate(X_train[val_index], y_train_cat[val_index])
    results = dict(zip(model.metrics_names,results))

    VALIDATION_ACCURACY.append(results['accuracy'])
    VALIDATION_LOSS.append(results['loss'])

    tensorflow.keras.backend.clear_session()

    fold_var += 1
	






pdb.set_trace()


model.save('test.hdf5')
#model.save('sandstone_50_epochs_catXentropy_acc_with_weights.hdf5')
############################################################
#Evaluate the model
	# evaluate model
#model.load_weights('test.hdf5')  

_, acc = model.evaluate(X_test, y_test_cat)
print("Accuracy is = ", (acc * 100.0), "%")
"""

###
#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
"""
##################################
#model = get_model()
#model.load_weights('sandstone_50_epochs_catXentropy_acc.hdf5')  
#model.load_weights('sandstone_50_epochs_catXentropy_acc_with_weights.hdf5')  
#model.load_weights('test.hdf5')  
#IOU

y_pred=model.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis=3)

##################################################

#Using built in keras function
from keras.metrics import MeanIoU
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_test, y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())


#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
#class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
#class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
#class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
#class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

#print("IoU for class1 is: ", class1_IoU)
#print("IoU for class2 is: ", class2_IoU)
#print("IoU for class3 is: ", class3_IoU)
#print("IoU for class4 is: ", class4_IoU)


#plt.imshow(train_images[0, :,:,0], cmap='gray')
#plt.imshow(train_masks[0], cmap='gray')
#######################################################################
#Predict on a few images
#model = get_model()
#model.load_weights('???.hdf5')  
pdb.set_trace()


import random
test_img_number = random.randint(0, len(X_test)-1)
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
test_img_norm=test_img[:,:,:,0][:,:,:,None]
test_img_input=np.expand_dims(test_img_norm, 0)
prediction = (model.predict(test_img_input))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]


plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img, cmap='jet')
plt.show()

pdb.set_trace()


test_img_number = random.randint(0, len(X_train)-1)
test_img = X_train[test_img_number]
ground_truth=y_train[test_img_number]
test_img_norm=test_img[:,:,:,0][:,:,:,None]
test_img_input=np.expand_dims(test_img_norm, 0)
prediction = (model.predict(test_img_input))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]


plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img, cmap='jet')
plt.show()



#####################################################################

#Predict on large image

#Apply a trained model on large image

"""
from patchify import patchify, unpatchify

large_image = cv2.imread('large_images/large_image.tif', 0)
#This will split the image into small images of shape [3,3]
patches = patchify(large_image, (128, 128), step=128)  #Step=256 for 256 patches means no overlap

predicted_patches = []
for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        print(i,j)
        
        single_patch = patches[i,j,:,:]       
        single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
        single_patch_input=np.expand_dims(single_patch_norm, 0)
        single_patch_prediction = (model.predict(single_patch_input))
        single_patch_predicted_img=np.argmax(single_patch_prediction, axis=3)[0,:,:]

        predicted_patches.append(single_patch_predicted_img)

predicted_patches = np.array(predicted_patches)

predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], 128,128) )

reconstructed_image = unpatchify(predicted_patches_reshaped, large_image.shape)
plt.imshow(reconstructed_image, cmap='gray')
#plt.imsave('data/results/segm.jpg', reconstructed_image, cmap='gray')

plt.hist(reconstructed_image.flatten())  #Threshold everything above 0

# final_prediction = (reconstructed_image > 0.01).astype(np.uint8)
# plt.imshow(final_prediction)

plt.figure(figsize=(8, 8))
plt.subplot(221)
plt.title('Large Image')
plt.imshow(large_image, cmap='gray')
plt.subplot(222)
plt.title('Prediction of large Image')
plt.imshow(reconstructed_image, cmap='jet')
plt.show()

"""