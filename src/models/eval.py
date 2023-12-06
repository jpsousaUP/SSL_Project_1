import os
import logging
from dotenv import find_dotenv, load_dotenv

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import sys

import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report


def model_evalue(model, X_test, y_test):
    """
        Evaluate the model with the test data
    """
    # evaluate model
    loss, acc = model.evaluate(X_test, y_test)

    # print loss and accuracy
    #print("loss:", loss)
    #print("accuracy:", acc)
    return loss, acc

def eval_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test, batch_size=32)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    #print("Confusion matrix:\n", cm)
    return y_pred, y_test, cm

def eval_classification_report(y_test, y_pred):
   cr = classification_report(y_test, y_pred)
   #print("Classification report:\n", cr)
   return cr


def main(project_dir, bag_name):
    # define the dataset path
    X_path = os.path.join(project_dir, "data", "processed", bag_name, "_X.npy")
    y_path = os.path.join(project_dir, "data", "processed", bag_name, "_y.npy")
    
    # Load test data only
    _, _, (X_test, y_test) = read_from_npy(X_path, y_path)
    
    # load model
    model_dir = os.path.join(project_dir, "logs", "20231127_123930/models/model_val_acc.tf")
    model = load_model(model_dir)
    
    # evaluate the model
    loss1, acc1 =  model_evalue(model, X_test, y_test)
    
    # confusion matrix
    y_pred1, y_test1, cm1 = eval_confusion_matrix(model, X_test, y_test)
    
    # classification report
    cr1 = eval_classification_report(y_test1, y_pred1) 
    
    # X_test shape is X=(1439, 320, 320, 2). put the first channel to zero
    # make a black image with the same shape as X_test[0]
    black_image = np.zeros(X_test[0].shape[0])
    #print(X_test[0].shape[0])
    
    X_test_new = np.copy(X_test)
    X_test_new[:, :, :, 0] = black_image
    # print first channel and second channel of image 1000
    #print(X_test_new[1000, :, :, 0], "\n")
    #print(X_test_new[1000, :, :, 1])
    
    # evaluate the model
    loss2, acc2 = model_evalue(model, X_test_new, y_test)
    y_pred2, y_test, cm2 = eval_confusion_matrix(model, X_test_new, y_test)
    cr2 = eval_classification_report(y_test, y_pred2)
    
    # compare both results
    print("Compare both results")
    print("loss:", loss1, loss2)
    print("accuracy:", acc1, acc2)
    print("Confusion matrix:\n", cm1, "\n", cm2)
    print("Classification report:\n", cr1, "\n", cr2)
    
    
    

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
