# -*- coding: utf-8 -*-
import os
import logging
from dotenv import find_dotenv, load_dotenv
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt

MELT_T = 1150
MAX_T = 2300
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
    feats = np.empty((timestamps.shape[0], 8),)

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
        xiris[i], xiris_img_8bit = preprocess_xiris(img_path_xiris, RESOLUTION, MAX_T)
        manta[i] = preprocess_manta(img_path_manta, RESOLUTION)
        
        # get the mask and contours
        xiris_mask, xiris_contours, feat = get_mp(xiris[i], xiris_img_8bit, MELT_T, MAX_T, plot=False, title=f"Xiris image {ts_xiris}")
        
        if i == 1000:
            # save xiris image as a png
            plt.imsave(os.path.join(project_dir, "reports/figures", f"xiris_{bag_name}-{ts_xiris}.png"), xiris[i], cmap="jet", vmin=0, vmax=1)
            get_mp(xiris[i], xiris_img_8bit, MELT_T, MAX_T, plot=True, title=f"Xiris image {ts_xiris}")
            
        # ts_xiris in feats[id, 0]
        feats[i, 0] = ts_xiris      
                
        # append the features
        feats[i, 1:] = feat    
        print(feat)

    """ # expand dimensions of xiris and manta
    xiris = np.expand_dims(xiris, axis=-1)
    manta = np.expand_dims(manta, axis=-1)

    # stack the images as 2 channel
    X = np.concatenate((xiris, manta), axis=-1)
    
    """
    # Append the last 2 columns of timestamps to y
    y = timestamps[:, 4:]
        
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

    # save xiris, manta, and features in seperad np files
    np.save(os.path.join(project_dir, "data/processed", bag_name, "_xiris.npy"), xiris)
    np.save(os.path.join(project_dir, "data/processed", bag_name, "_manta.npy"), manta)
    np.save(os.path.join(project_dir, "data/processed", bag_name, "_feats.npy"), feats)
    #np.save(os.path.join(project_dir, "data/processed", bag_name, "_X.npy"), X)
    np.save(os.path.join(project_dir, "data/processed", bag_name, "_y.npy"), y)
    
    # print shapes
    print("xiris shape: ", xiris.shape)
    print("manta shape: ", manta.shape)
    print("feats shape: ", feats.shape)
    #print("X shape: ", X.shape)
    print("y shape: ", y.shape)
    

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


def preprocess_xiris(img_path, resolution=320, max_temp=2300):
    """
    Preprocess xiris images
    """
    
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = img /10 # temperatre in ºC 
    # change to numpy float32 from FLOAT64git
    img = np.float32(img)
    # normalize the image using numpy
    img = img/(max_temp)
    # clip the image to 25000 max
    np.where(img > 1, 1, img)
    
    
    #img = np.clip(img, 0, 1) # cv2.normalize(cropped, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16U) #np.clip(cropped, 0, 1) #
    
    
    # crop image in the center with 300 x300 pixels
    height, width = img.shape[:2]
    start_x = int((width - resolution) / 2) -20
    start_y = int((height - resolution) / 2)-60
    cropped = img[start_y:start_y+resolution, start_x:start_x+resolution]
    
    # get 8bit image
    cropped_8bit = cropped*255
    cropped_8bit = np.uint8(cropped_8bit)    
    return cropped, cropped_8bit


# define a function to filter the contours
def filter_contours(raw_contours, min_contour_size):
    # Create a list to store the filtered contours
    filtered_contours = []
    # Iterate over the contours
    for contour in raw_contours:
        # Check if the contour is larger than the minimum size
        if cv2.contourArea(contour) > min_contour_size:
            # Add the contour to the list of filtered contours
            filtered_contours.append(contour)
    return filtered_contours


def get_mp(img, img_8bit, thresh_temperature_1, thresh_temperature_2, min_contour_size=10, plot=False, title="Original image"):
    # create a copy of the original image
    img_copy = img.copy() *MAX_T
    
    
    # Set threshold values to pixel values
    lower_threshold = thresh_temperature_1 / MAX_T * 255 #2500 
    upper_threshold = thresh_temperature_2 / MAX_T * 255 #2500 

    # Apply threshold
    img_8bit_blur = cv2.medianBlur(img_8bit, 3)
    
    # create a mask
    mask = cv2.inRange(img_8bit_blur, lower_threshold, upper_threshold)

    # find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS) 
    filtered_contours = filter_contours(contours, min_contour_size)
    filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)

    # create a mask from the largest contour
    mask_final = np.zeros_like(img_8bit)
    cv2.drawContours(mask_final, filtered_contours, -1, (255, 255, 255), -1)
    # close mask 
    kernel = np.ones((3,3), np.uint8)
    #mask_final = cv2.dilate(mask_final, kernel, iterations=1)
    #mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel, iterations=10)
    mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel, iterations=10)

    # final contour
    contours_final, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # create a cropped image from the largest contour
    x,y,L,H = cv2.boundingRect(mask_final)#filtered_contours[-1])
    # draw the rectangle on the original image
    cv2.rectangle(img_copy, (x, y), (x+L, y+H), (0, 0, 255), 1)
    
    feats = get_features(img, mask_final)

    if plot:
        # multiply img by MAX_T to get the original temperature
        img_copy2 = img * MAX_T
        
        # make a "title" wjere displays the min and max temperature
        title = f"Original image: min={np.min(img_copy2):.1f}ºC, max={np.max(img_copy2):.1f}ºC"
        
        mask_1350 = np.where(img_copy2 > 1350, 1, 0)
        # plot all the image steps
        fig, axs = plt.subplots(1, 3, figsize=(10, 5), dpi=200)
        axs[0].imshow(img_copy2, cmap="jet", vmin=0, vmax=MAX_T)
        axs[0].set_title(f"{title}")
       # plot a drwaing of the contours on the img
        axs[1].imshow(cv2.drawContours(img.copy(), contours_final, -1, (0, 200, 0), 1), cmap="jet", vmin=0, vmax=1)# MAX_T)
        axs[1].set_title("Contours")
       
        axs[2].imshow(img_copy, cmap="jet", vmin=0, vmax=MAX_T)
        axs[2].set_title("Contour image")
        # overlay with 50% transparency 
        axs[2].imshow(mask_final, alpha=0.2)
        axs[2].imshow(mask_1350, alpha=0.5)
        # add color bar on axis 0
        fig.colorbar(axs[0].imshow(img_copy2, cmap="jet", vmin=0, vmax=MAX_T, alpha=1.0), ax=axs[0], orientation="horizontal", pad=0.01)

        # tur off all axis
        for ax in axs:
            ax.axis("off")
        plt.tight_layout()   
        
        # save plot in reports folder
        plt.savefig(os.path.join(project_dir, "reports/figures", f"{bag_name}-{title}.png"), dpi=500)
        
    # mask_final as 0 and 1
    mask_final = np.where(mask_final > 0, 1, 0)
    return mask_final, contours_final, feats

def get_features(img, mask):
    '''
    img is the original image (cropped or not)
    mask is a 8bit image with 0 and 255
    
    FEATURES: are calculated inside the mask !!!
    '''
    # calculate manual features
    area = np.sum(mask)
    
    # get the max and min temperature in the original image inside the mask
    masked_img = np.where(mask==255, img, np.nan)
    max_temp = np.nanmax(masked_img)* MAX_T
    min_temp = np.nanmin(masked_img)* MAX_T
    mean_temp = np.nanmean(masked_img)*MAX_T
    std_temp = np.nanstd(masked_img)*MAX_T
    
    # get temperatures with onli one decimal
    max_temp = np.round(max_temp, 1)
    min_temp = np.round(min_temp, 1)
    mean_temp = np.round(mean_temp, 1)
    std_temp = np.round(std_temp, 1)
    
    """ max_temp = np.max(img)* MAX_T
    min_temp = np.min(img)* MAX_T
    mean_temp = np.mean(img)* MAX_T
    std_temp = np.std(img)* MAX_T """
    
    x,y,L,H = cv2.boundingRect(mask)
    
    feats = L, H, area, max_temp, min_temp, mean_temp, std_temp
    return feats




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
