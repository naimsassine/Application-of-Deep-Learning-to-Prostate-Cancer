from common_functions import *
from deep_learning_model import *
from functools import partial

# Even though here its a complete different task from prostate segmentation, I used the same structure of the code
# So the structure of the code is inspired by https://github.com/mirzaevinom/promise12_segmentation even though it has
# nothing to do with what the code in the source does, I used the same architecture

# Please note that this file isn't used in the latest version of my master thesis (its not up to date), 
# simply because I directly tested lesion detection and
# classification at the same time, and didn't bother going through detection first, then both

def training_model_with_da():

    prepare_data_lesion_d()

    X_train, y_train, X_val, y_val = load_only_training()
    img_rows = X_train.shape[1]
    img_cols =  X_train.shape[2]
    img_depth = X_train.shape[3]

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

    image_datagen = ImageDataGenerator(**data_gen_args) 
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    batch_size = 32
    n_imgs = 50000
    seed = 2
    image_datagen.fit(X_train, seed=seed) 
    mask_datagen.fit(y_train, seed=seed)
    
    image_generator = image_datagen.flow(X_train, batch_size=batch_size, seed=seed)
    mask_generator = mask_datagen.flow(y_train, batch_size=batch_size, seed=seed)
    train_generator = zip(image_generator, mask_generator)
    model = UNet((img_rows, img_cols, img_depth), start_ch=8, depth=4, batchnorm=True, dropout=0.5, maxpool=True, residual=True)
    #model.load_weights('data/detection/weights_with_augmentation.h5')

    model.summary()
    model_checkpoint = ModelCheckpoint(
        'data/detection/weights_with_augmentation.h5', monitor='val_loss', save_best_only=True)

    call_backs = [model_checkpoint]
    #call_backs.append(EarlyStopping(monitor='loss', min_delta=0.00001, patience=5) )

    model.compile(optimizer=Adam(lr=0.0001), loss=dice_coef_loss, metrics=[dice_coef])

    history = model.fit_generator(train_generator,
                        steps_per_epoch=n_imgs//batch_size,
                        #steps_per_epoch=5,
                        epochs=120,
                        #epochs=5,
                        verbose=1,
                        shuffle=True,
                        validation_data=(X_val, y_val),
                        callbacks=call_backs,
                        use_multiprocessing=True)
    
    np.save('data/detection/history_with_augmentation.npy',history.history)

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
    from matplotlib import pyplot as plt

    plt.plot(epochs, acc, 'y', label='Training DSC')
    plt.plot(epochs, val_acc, 'r', label='Validation DSC')
    plt.title('Training and validation DSC')
    plt.xlabel('Epochs')
    plt.ylabel('DSC')
    plt.legend()
    plt.show()

def training_model_without_da():
    prepare_data_lesion_d()

    X_train, y_train, X_val, y_val = load_only_training()
    img_rows = X_train.shape[1]
    img_cols =  X_train.shape[2]
    img_depth = X_train.shape[3]

    model = UNet((img_rows, img_cols, img_depth), start_ch=8, depth = 4, batchnorm=True, dropout=0.5, maxpool=True, residual=True)
    # model.load_weights('/data/detection/weights_no_da.h5')

    model.summary()
    model_checkpoint = ModelCheckpoint(
        'data/detection/weights_no_da.h5', monitor='val_loss', save_best_only=True)

    call_backs = [model_checkpoint]
    #call_backs.append(EarlyStopping(monitor='loss', min_delta=0.00001, patience=5))

    model.compile(optimizer=Adam(lr=0.0001), loss=dice_coef_loss, metrics=[dice_coef])


  
    history = model.fit(X_train, y_train, batch_size = 32, epochs = 120, callbacks=call_backs, verbose=1, validation_data=(X_val, y_val))
    np.save('data/detection/history_no_augmentation.npy',history.history)

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


training_model_without_da()

def testing_model_no_augmentation():
    model = get_model_d(128, 128, 4, 'data/detection/weights_no_da.h5')
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    y_pred = model.predict(X_test, verbose=1, batch_size=128)

    check_predictions("test", y_pred, var="detection")

    model.layers[-1].activation = None
    
    # grad-cam
    img_3d_array = np.expand_dims(X_train[0], axis=0)
    preds = model.predict(img_3d_array)
    last_conv_layer_name = "conv2d_61" # taken from model.summary()
    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(img_3d_array, model, last_conv_layer_name, pred_index=1)
    # Display heatmap
    plt.matshow(heatmap)
    plt.show()

    # drawing some cases
    make_plots(X_test, y_test, y_pred)

def testing_model_augmentation():
    model = get_model_d(128, 128, 4, 'data/detection/weights_with_augmentation.h5')
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    y_pred = model.predict(X_test, verbose=1, batch_size=128)

    check_predictions("test", y_pred, var = "detection")


    model.layers[-1].activation = None
    
    # grad-cam
    img_3d_array = np.expand_dims(X_train[0], axis=0)
    preds = model.predict(img_3d_array)
    last_conv_layer_name = "conv2d_61" # taken from model.summary()
    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(img_3d_array, model, last_conv_layer_name, pred_index=1)
    # Display heatmap
    plt.matshow(heatmap)
    plt.show()

    # drawing some cases
    make_plots(X_test, y_test, y_pred)

