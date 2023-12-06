import os
import logging
from dotenv import find_dotenv, load_dotenv

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Define paths
#dataset_path = r"TF_multimodal_1/data"
#path_X = dataset_path + "/merged"
#path_csv = dataset_path + "/dataset.csv"


def main(project_dir, bag_name):
    
    # get paths of X and y from data/processed
    X_path = os.path.join(project_dir, "data", "processed", bag_name, "_X.npy")
    y_path = os.path.join(project_dir, "data", "processed", bag_name, "_y.npy")
    
    # read X and y from npy
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = read_from_npy(X_path, y_path)
    # print shapes
    print("X_train, y_train shape:", X_train.shape, y_train.shape)
    print("X_val, y_val shape:", X_val.shape, y_val.shape)
    print("X_test, y_test shape:", X_test.shape, y_test.shape)
    
    # min and max of X_test and y_test
    print("X_test min and max:", np.min(X_test), np.max(X_test))
    print("y_test min and max:", np.min(y_test), np.max(y_test))
 
    
def read_from_npy(X_path, y_path):
    X = np.load(X_path)
    y = np.load(y_path)
    # get only the first column of y # laser power
    y = y[:, 0]
    # transform y to one-hot encoding
    N_CLASSES = len(np.unique(y))
    _, y_unique = np.unique(y, return_inverse=True)
    y_encoded = tf.keras.utils.to_categorical(y_unique, num_classes=N_CLASSES)
    
    # split in train, val and test
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)
    
    # print shape of X_train, X_test, y_train, y_test
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation shapes: X={X_val.shape}, y={y_val.shape}")
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")
    
    # get number of classes of y
    N_CLASSES = y_val.shape[-1]
    N_CHANNELS = X_train.shape[-1]
    RESOLUTION = X_train.shape[1]
    print("Number of classes:", N_CLASSES)
    print("Number of channels:", N_CHANNELS)
    print("Resolution:", RESOLUTION)


    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    
    # print project path from .env
    bag_name = os.getenv("BAG_NAME")
    # print project path from .env
    project_dir = os.getenv("PROJECT_DIR")

    main(project_dir, bag_name)



    