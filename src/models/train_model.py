import os
import logging
from dotenv import find_dotenv, load_dotenv


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import datetime
import sys



# Tensorflow imports
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
from keras.preprocessing.image import ImageDataGenerator    
from keras.callbacks import TensorBoard, ModelCheckpoint


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # print
    print('GPU memory growth:', tf.config.experimental.get_memory_growth(physical_devices[0]))




def train_simpleCNN(X_train, y_train, X_val, y_val, 
                    loss, optimizer, metrics, epochs, batch_size, data_augmentation):

    # build the model
    model = build_simpleCNN_model(X_train.shape[1], X_train.shape[-1])
    # print model summary
    model.summary()

    # # define ModelCheckpoint callback
    now = datetime.datetime.now()
    
    # define log dir: project_name + /logs
    log_dir = os.path.join(project_dir, "logs", now.strftime("%Y%m%d_%H%M%S"))
    
    # make dir if not exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # define model logs
    model_cbk_dir = os.path.join(log_dir, 'models')
    model_name_1 = os.path.join(model_cbk_dir, "model_val_loss.tf")
    model_name_2 = os.path.join(model_cbk_dir, "model_val_acc.tf")
    
    # make dir if not exist
    if not os.path.exists(model_cbk_dir):
        os.makedirs(model_cbk_dir)

    # compile the model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    # define TensorBoard callback
    tb_callback = TensorBoard(log_dir= log_dir, histogram_freq=1, write_graph=True)

    # define ModelCheckpoint callback
    mc_callback_1 = ModelCheckpoint(model_name_1, monitor='val_loss', mode='min', save_best_only=True)
    mc_callback_2 = ModelCheckpoint(model_name_2, monitor='val_accuracy', mode='max', save_best_only=True)
    
    if data_augmentation:
        # define data preparation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True)
        # fit parameters from data
        datagen.fit(X_train)
        # configure batch size and retrieve one batch of images
        train_gen = datagen.flow(X_train, y_train, batch_size=batch_size)
        # train the model
        history = model.fit(train_gen, 
                            steps_per_epoch=len(X_train) / batch_size, 
                            epochs=epochs, 
                            validation_data=(X_val, y_val),
                            workers=4,
                            use_multiprocessing=True,
                            callbacks=[tb_callback, mc_callback_1, mc_callback_2])
    else:
        # train the model
        history = model.fit(X_train, y_train, 
                            validation_data=(X_val, y_val),
                            epochs=epochs, batch_size=batch_size,
                            callbacks=[tb_callback, mc_callback_1, mc_callback_2])


def main(project_dir, bag_name):
    
    # define the dataset path
    X_path = os.path.join(project_dir, "data", "processed", bag_name, "_X.npy")
    y_path = os.path.join(project_dir, "data", "processed", bag_name, "_y.npy")
    
    # Load train data (train and validation only)
    (X_train, y_train), (X_val, y_val), _ = read_from_npy(X_path, y_path)

    LOSS = "categorical_crossentropy"
    OPTIMIZER = "adam"
    METRICS = ["accuracy"]
    EPOSCHS = 10
    BATCH_SIZE = 32
    DATA_AUGMENTATION = False
    
    # test the function 
    train_simpleCNN(X_train, y_train, X_val, y_val,
                    loss=LOSS, 
                    optimizer=OPTIMIZER, 
                    metrics=METRICS, 
                    epochs=EPOSCHS, 
                    batch_size=BATCH_SIZE, 
                    data_augmentation=DATA_AUGMENTATION)


if __name__ == '__main__':
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    
    # print project path from .env
    bag_name = os.getenv("BAG_NAME")
    # print project path from .env
    project_dir = os.getenv("PROJECT_DIR")
    # insert sys src 
    sys.path.insert(0, os.path.join(project_dir, "src"))
    # Local imports
    from features.build_features import read_from_npy
    from model_builder import build_simpleCNN_model

    main(project_dir, bag_name)
