import cv2
import numpy as np  
import matplotlib.pyplot as plt

from mcap_ros2.decoder import DecoderFactory
from mcap.reader import make_reader
import os
import csv
from dotenv import load_dotenv, find_dotenv
 
def main(project_dir, bag_name):
    """
        Extract the images from the mcap file and save them into a folders
        Attention to the project_dir path
        
        Runs data processing scripts to turn raw data from (../raw) into
        interim data ready to be pre-processed analyzed (saved in ../interim).
    """
    print("#########################")
    print("###", "extract_mcap.py", "####")
    print("#########################")
    print('###', project_dir, '###')
    print("#########################")
    bag_name = os.getenv('BAG_NAME')
    print("Bag name: ", bag_name)

    # bag path
    bag_path = os.path.join(project_dir, 'data', 'raw', bag_name)
    # list all mcap files
    mcap_files = [os.path.join(bag_path, f) for f in os.listdir(bag_path) if f.endswith('.mcap')]
    mcap_files.sort()
    print("MCAP files: ", mcap_files)


    itermin_data_path = os.path.join(project_dir, 'data', 'interim', bag_name)

    # in intermin folder, create a folders for xiris, manta and plc if this folders do not exist
    if not os.path.exists(os.path.join(itermin_data_path, 'xiris')):
        os.makedirs(os.path.join(itermin_data_path, 'xiris'))
    if not os.path.exists(os.path.join(itermin_data_path, 'manta')):
        os.makedirs(os.path.join(itermin_data_path, 'manta'))
    if not os.path.exists(os.path.join(itermin_data_path, 'plc')):
        os.makedirs(os.path.join(itermin_data_path, 'plc'))

    id = 0
    xiris_stamp = 0
    manta_stamp = 0
    
    for mcap_file in mcap_files:
        with open(mcap_file, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for schema, channel, message, ros_msg in reader.iter_decoded_messages():
                print(f"{channel.topic} {schema.name} [{message.log_time}")
                if channel.topic == "/xiris/compressed":
                    # save the ros2 image into xiris folder with the timestamp as name
                    xiris_folder = os.path.join(itermin_data_path, 'xiris', )
                    # save the ros2 image
                    save_compressed_image(ros_msg.data, xiris_folder, str(message.log_time) + ".png")
                    # save the timestamp
                    xiris_stamp = message.log_time
                elif channel.topic == "/manta/compressed":
                    # save the ros2 image into manta folder with the timestamp as name
                    manta_folder = os.path.join(itermin_data_path, 'manta')
                    # save the ros2 image
                    save_compressed_image(ros_msg.data, manta_folder, str(message.log_time) + ".png")
                    # save the timestamp
                    manta_stamp = message.log_time
                elif channel.topic == "/ads":
                    # save the plc data to a csv file
                    plc_csv_path = os.path.join(itermin_data_path, 'plc', 'plc.csv')
                    plc_msg_to_csv(ros_msg, message.log_time, plc_csv_path)     
                
                id += 1 
                data = {
                    'id': id,
                    'time_stamp': message.log_time,
                    'xiris_stamp': xiris_stamp,
                    'manta_stamp': manta_stamp,
                }
                if manta_stamp > 0 and xiris_stamp > 0 :
                    # append the data to the csv file
                    timestamps_path = os.path.join(itermin_data_path, 'timestamps.csv')
                    # if the file does not exist, create it and add the header
                    if not os.path.exists(timestamps_path):
                        with open(timestamps_path, 'w', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=data.keys())
                            writer.writeheader()
                    with open(timestamps_path, 'a+', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=data.keys())
                        writer.writerow(data)    
                        
                    
# function to read the ros2 msg and save the image
def save_compressed_image(msg_data, folder, image_name):
    # Convert the byte array to a numpy array
    nparr = np.frombuffer(msg_data, np.uint8)
    # Decode the image
    image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    # Save the image
    cv2.imwrite(os.path.join(folder, image_name), image)
    
def plc_msg_to_csv(plc_msg, time_stamp, csv_path):
    # create a big dictionary with the plc_dict and robot_dict and the time_stamp
    data = {
        'time_stamp': time_stamp,
        'program_running': plc_msg.plc.program_running,
        'laser_on': plc_msg.plc.laser_on,
        'laser_power_out': plc_msg.plc.laser_power_out,
        'laser_power_in': plc_msg.plc.laser_power_in,
        'laser_freq_out': plc_msg.plc.laser_freq_out,
        'laser_program': plc_msg.plc.laser_program,
        'act_nozzlegas': plc_msg.plc.act_nozzlegas,
        'pf1_start_process': plc_msg.plc.pf1_start_process,
        'pf1_act_speed': plc_msg.plc.pf1_act_speed,
        'pf1_act_carr_gas': plc_msg.plc.pf1_act_carr_gas,
        'pf1_act_feed': plc_msg.plc.pf1_act_feed,
        'pf1_act_flowwatch': plc_msg.plc.pf1_act_flowwatch,
        'pf2_start_process': plc_msg.plc.pf2_start_process,
        'pf2_act_speed': plc_msg.plc.pf2_act_speed,
        'pf2_act_carr_gas': plc_msg.plc.pf2_act_carr_gas,
        'pf2_act_feed': plc_msg.plc.pf2_act_feed,
        'pf2_act_flowwatch': plc_msg.plc.pf2_act_flowwatch,
        'base_number': plc_msg.robot.base_number,
        'pos_x': plc_msg.robot.pos_x,
        'vel_x': plc_msg.robot.vel_x,
        'acc_x': plc_msg.robot.acc_x,
        'pos_y': plc_msg.robot.pos_y,
        'vel_y': plc_msg.robot.vel_y,
        'acc_y': plc_msg.robot.acc_y,
        'pos_z': plc_msg.robot.pos_z,
        'vel_z': plc_msg.robot.vel_z,
        'acc_z': plc_msg.robot.acc_z,
        'pos_a': plc_msg.robot.pos_a,
        'vel_a': plc_msg.robot.vel_a,
        'acc_a': plc_msg.robot.acc_a,
        'pos_b': plc_msg.robot.pos_b,
        'vel_b': plc_msg.robot.vel_b,
        'acc_b': plc_msg.robot.acc_b,
        'pos_c': plc_msg.robot.pos_c,
        'vel_c': plc_msg.robot.vel_c,
        'acc_c': plc_msg.robot.acc_c,
        'robot_is': plc_msg.robot.robot_is,
        'robot_it': plc_msg.robot.robot_it,
    }

    # append the data to the csv file
    with open(csv_path, 'a+', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        writer.writerow(data)
    return data

  
if __name__ == '__main__':
    
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    
    # print project path from .env
    bag_name = os.getenv("BAG_NAME")
    # print project path from .env
    project_dir = os.getenv("PROJECT_DIR")

    main(project_dir, bag_name)