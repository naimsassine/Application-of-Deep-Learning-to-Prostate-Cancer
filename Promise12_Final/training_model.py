# Here is where the training happens

from pdb import set_trace
from pre_processing_data import *
from deep_learning_model import *
from exp_and_results import *
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from functools import partial

# I inspired myself by the way that https://github.com/mirzaevinom/promise12_segmentation/ was adapting to accomplish some data augmentation
# but for the rest of the code, its typical tensorflow/keras training a model, saving the weights for the best validation score, and plotting some 
# curves

# inspired by : https://github.com/mirzaevinom/promise12_segmentation/blob/master/codes/metrics.py 
def training_first_test():

    convert_img_to_array()
    X_train, y_train, X_val, y_val = load_only_training()
    img_rows = X_train.shape[1]
    img_cols =  X_train.shape[2]
    img_depth = X_train.shape[3]


    model = UNet((img_rows, img_cols,1), start_ch=8, depth=5, batchnorm=True, dropout=0.5, maxpool=True, residual=True)
    # model.load_weights('data/weights_no_augmentation.h5')

    model.summary()
    model_checkpoint = ModelCheckpoint('data/weights_no_augmentation.h5', monitor='val_loss', save_best_only=True)

    call_backs = [model_checkpoint]

    model.compile(optimizer=Adam(lr=0.0001), loss=dice_coef_loss, metrics=[dice_coef])


    # the batch size could be a paramter that can vary and affect the results. Will I have time to test different batch sizes??
    history = model.fit(X_train, y_train, batch_size = 32, epochs = 120, callbacks=call_backs, verbose=1, validation_data=(X_val, y_val))
    np.save('data/history_no_augmentation.npy',history.history)
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


    acc = history.history['dice_coef']
    val_acc = history.history['val_dice_coef']
    plt.plot(epochs, acc, 'y', label='Training DSC')
    plt.plot(epochs, val_acc, 'r', label='Validation DSC')
    plt.title('Training and validation DSC')
    plt.xlabel('Epochs')
    plt.ylabel('DSC')
    plt.legend()
    plt.show()

# inspired by : https://github.com/mirzaevinom/promise12_segmentation/blob/master/codes/metrics.py 
def train_with_data_augmentation():
    convert_img_to_array()

    X_train, y_train, X_val, y_val = load_only_training()

    img_rows = X_train.shape[1]
    img_cols =  X_train.shape[2]

    x, y = np.meshgrid(np.arange(img_rows), np.arange(img_cols), indexing='ij')
    elastic = partial(elastic_transform, x=x, y=y, alpha=img_rows*1.5, sigma=img_rows*0.07)

    data_gen_args = dict(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=[1, 1.2],
        fill_mode='constant',
        preprocessing_function=elastic)

    data_generator_imgs = ImageDataGenerator(**data_gen_args) 
    data_generator_masks = ImageDataGenerator(**data_gen_args)

    data_generator_imgs.fit(X_train, seed=2) 
    data_generator_masks.fit(y_train, seed=2)
    

    image_generator = data_generator_imgs.flow(X_train, batch_size=32, seed=2)
    mask_generator = data_generator_masks.flow(y_train, batch_size=32, seed=2)

    train_generator = zip(image_generator, mask_generator)
    model = UNet((img_rows, img_cols,1), start_ch=8, depth=5, batchnorm=True, dropout=0.5, maxpool=True, residual=True)

    # model.load_weights('data/weights_with_augmentation.h5')

    model.summary()
    from tensorflow.keras.utils import plot_model
    # plot_model(model, to_file='model.png')
    model_checkpoint = ModelCheckpoint(
        'data/weights_with_augmentation.h5', monitor='val_loss', save_best_only=True)

    call_backs = [model_checkpoint]
    #call_backs.append( EarlyStopping(monitor='loss', min_delta=0.00001, patience=5) )

    model.compile(optimizer=Adam(lr=0.001), loss=dice_coef_loss, metrics=[dice_coef])

    number_of_images = 30000 # since the data is generated at each epoch, this represents the number of images we want to train in for each epoch
    batch_size = 32
    
    history = model.fit_generator(train_generator,
                        steps_per_epoch=number_of_images//batch_size,
                        epochs=60,
                        verbose=1,
                        shuffle=True,
                        validation_data=(X_val, y_val),
                        callbacks=call_backs,
                        use_multiprocessing=False)
    np.save('data/history_with_augmentation.npy',history.history)

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


    acc = history.history['dice_coef']
    val_acc = history.history['val_dice_coef']
    plt.plot(epochs, acc, 'y', label='Training DSC')
    plt.plot(epochs, val_acc, 'r', label='Validation DSC')
    plt.title('Training and validation DSC')
    plt.xlabel('Epochs')
    plt.ylabel('DSC')
    plt.legend()
    plt.show()


if __name__ == "__main__": 
    train_with_data_augmentation()
