# -*- coding: utf-8 -*-
import os
import logging
from dotenv import find_dotenv, load_dotenv
import csv
import numpy as np
import cv2


RESOLUTION = 320

def main(project_dir, bag_name):
    """ Runs data processing scripts to turn raw data from (../interim) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    # timestamps_csv is int interim folder
    timestamps_csv = os.path.join(project_dir, "data/interim", bag_name, "timestamps.csv")

    # read csv with numpy as integers
    timestamps = np.genfromtxt(timestamps_csv, delimiter=',', dtype=np.int64, skip_header=1)
    #print(timestamps.shape)
    #print(timestamps[0:10, :])
    
    # Add machine parameters  to the dataframe
    id_sep = [0]
    # for all xiris stamps if delta T is more then 10**8 append to id_sep
    for i in range(1, timestamps.shape[0]):
        # subtract the current timestamp from the previous one in column 1
        delta = timestamps[i, 1] - timestamps[i-1, 1]
        # if delta is more then 10**8 append to id_sep
        if delta > 10**8:
            id_sep.append(i)
    # apend the last index
    id_sep.append(timestamps.shape[0])
    print(id_sep)

    # Process parameters tested
    power = [500, 1250, 2000, 2750]
    velocity = [5, 10, 15]
    pw = []
    vl = []

    for p in power:
        for v in velocity:
            pw.append(p)
            vl.append(v)
    print(pw, vl)

    # add power and velocity to the timestamps using id_sep as index
    # add 2 empty columns to timestamps as int64
    timestamps = np.hstack((timestamps, np.zeros((timestamps.shape[0], 2), dtype=np.int64)))

    # add power and velocity to timestamps using id_steo as index
    for i in range(len(id_sep)-1):
        # add power and velocity to timestamps using id_sep as index as integers
        timestamps[id_sep[i]:id_sep[i+1], 4] = int(pw[i])
        timestamps[id_sep[i]:id_sep[i+1], 5] = int(vl[i])

    # save timestamps to csv
    timestamps_csv = os.path.join(project_dir, "data/processed", bag_name) #
    # create folder if not exist
    if not os.path.exists(timestamps_csv):
        os.makedirs(timestamps_csv)
    # add file name
    timestamps_csv = os.path.join(timestamps_csv, "timestamps.csv")
    np.savetxt(timestamps_csv, timestamps, delimiter=",", fmt='%d')
    
    # GET INPUTS (images)
    # Preallocate numpy arrays for efficiency
    xiris = np.empty((timestamps.shape[0], RESOLUTION, RESOLUTION))
    manta = np.empty((timestamps.shape[0], RESOLUTION, RESOLUTION))

    # Load xiris and manta images with timestamps (columns 2 and 3)
    for i in range(timestamps.shape[0]): #range(100): #
        # Get the timestamp for xiris and manta
        ts_xiris = timestamps[i, 2]
        ts_manta = timestamps[i, 3]
        
        # define image paths for each camera
        img_path_xiris = os.path.join(project_dir, "data/interim", bag_name, "xiris")
        img_path_manta = os.path.join(project_dir, "data/interim", bag_name, "manta")
        
        # create folder if does not exist
        if not os.path.exists(img_path_xiris):
            os.makedirs(img_path_xiris)
        if not os.path.exists(img_path_manta):
            os.makedirs(img_path_manta)
            
        # add file name
        img_path_xiris = os.path.join(img_path_xiris, str(ts_xiris) + ".png")
        img_path_manta = os.path.join(img_path_manta, str(ts_manta) + ".png")
    

        # Load the images using preprocess_xiris and preprocess_manta
        # load in parallel
        xiris[i], manta[i] = preprocess_xiris(img_path_xiris, RESOLUTION), preprocess_manta(img_path_manta, RESOLUTION)
        # load in parallel

    # expand dimensions of xiris and manta
    xiris = np.expand_dims(xiris, axis=-1)
    manta = np.expand_dims(manta, axis=-1)

    # stack the images as 2 channel
    X = np.concatenate((xiris, manta), axis=-1)

    # Append the last 2 columns of timestamps to y
    y = timestamps[:, 4:]

    # Print shapes
    print("X shape: ", X.shape)
    print("y shape: ", y.shape)

        
    # create a dict with dataset info
    dataset_info = {
                    "description": "Images from xiris and manta cameras with power and velocity parameters. Channel 0 are xiris images and Channel 1 are manta images; Output: power and velocity",
                    "timestamps_csv": timestamps_csv,
                    "RESOLUTION": RESOLUTION,
                    "id_sep": id_sep,
                    "power": power,
                    "velocity": velocity,
                    "timestamps": timestamps}        
    dataset_info_csv = os.path.join(project_dir, "data/processed", bag_name, "dataset_info.csv")
    with open(dataset_info_csv, 'w+', newline='', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        for key, value in dataset_info.items():
            csv_writer.writerow([key, value])

    # save X, y with npy
    np.save(os.path.join(project_dir, "data/processed", bag_name, "_X.npy"), X)
    np.save(os.path.join(project_dir, "data/processed", bag_name, "_y.npy"), y)
    

def preprocess_manta(img_path, resolution=320):
    """
    Preprocess manta images
    """
    
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    # crop image in the center with 300 x300 pixels
    height, width = img.shape[:2]
    #print(height, width)
    
    start_x = int((width - resolution) / 2) -20
    start_y = int((height - resolution) / 2)
    cropped = img[start_y:start_y+resolution, start_x:start_x+resolution]

    # change to numpy float32
    cropped = np.float16(cropped)
    # normalize the image using numpy
    cropped = cropped/255
    
    return cropped


def preprocess_xiris(img_path, resolution=320):
    """
    Preprocess xiris images
    """
    
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    # crop image in the center with 300 x300 pixels
    height, width = img.shape[:2]
    #print(height, width)
    
    start_x = int((width - resolution) / 2) -20
    start_y = int((height - resolution) / 2)-60
    cropped = img[start_y:start_y+resolution, start_x:start_x+resolution]
    
    # change to numpy float32
    cropped = np.float16(cropped)
    # clip the image to 25000 max
    cropped = np.clip(cropped, 0, 25000)
    # normalize the image using numpy
    cropped = cropped/25000
    
    
    return cropped


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
