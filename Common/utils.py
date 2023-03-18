import numpy as np
import os
import shutil
import cv2
import torch
import select
import sys
import termios
import tty
from math import log10, floor
import psutil


##########
# Angles #
##########

def compute_absolute_angle_difference(angle_1, angle_2):
    angle = np.pi - np.abs(np.abs(angle_1 - angle_2) - np.pi)
    return angle


#######################
# Files / Directories #
#######################


# Function to create an empty directory
def create_or_clear_directory(dir_path):
    if not does_directory_exist(dir_path):
        create_directory(dir_path)
    else:
        delete_directory_contents(dir_path)


# Function to create a directory
def create_directory(dir_path):
    os.makedirs(dir_path)


# Function to create a directory if it doesn't already exist
def create_directory_if_not_exist(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def create_directory_if_none(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


# Function to check if a directory already exists
def does_directory_exist(dir_path):
    if os.path.isdir(dir_path):
        return True
    else:
        return False


# Function to delete all files and directories in a given directory
def delete_directory_contents(dir_path):
    if os.path.isdir(dir_path):
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)


def get_all_directories_in_directory(directory_path):
    directories = []
    for path in os.listdir(directory_path):
        full_path = os.path.join(directory_path, path)
        if os.path.isdir(full_path):
            directories.append(full_path)
    return directories


def get_all_files_in_directory(directory_path):
    files = []
    for path in os.listdir(directory_path):
        full_path = os.path.join(directory_path, path)
        if os.path.isfile(full_path):
            files.append(full_path)
    return files


def get_num_directories_in_directory(dir_path):
    all_dirs = get_all_directories_in_directory(dir_path)
    num_dir = len(all_dirs)
    return num_dir


def get_num_files_in_directory(dir_path):
    all_files = get_all_files_in_directory(dir_path)
    num_files = len(all_files)
    return num_files


#############################
# Loading and saving images #
#############################


def save_rgb_float_image(image_path, rgb_image):
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    uint8_bgr_image = (255 * bgr_image).astype(np.uint8)
    cv2.imwrite(image_path, uint8_bgr_image)


#######################
# Number manipulation #
#######################


def round_to_sig_fig(x, sig):
    return round(x, sig-int(floor(log10(abs(x))))-1)


####################
# Colours #
####################


def get_colour(colour_index):
    colour_index = colour_index % 16
    if colour_index == 0:
        return [255, 0, 0]
    elif colour_index == 1:
        return [0, 255, 0]
    elif colour_index == 2:
        return [0, 0, 255]
    elif colour_index == 3:
        return [255, 255, 0]
    elif colour_index == 4:
        return [0, 255, 255]
    elif colour_index == 5:
        return [255, 0, 255]
    elif colour_index == 6:
        return [0, 127, 0]
    elif colour_index == 7:
        return [0, 255, 127]
    elif colour_index == 8:
        return [255, 0, 127]
    elif colour_index == 9:
        return [127, 0, 127]
    elif colour_index == 10:
        return [255, 127, 255]
    elif colour_index == 11:
        return [255, 127, 0]
    elif colour_index == 12:
        return [255, 127, 127]
    elif colour_index == 13:
        return [255, 127, 255]
    elif colour_index == 14:
        return [255, 255, 0]
    elif colour_index == 15:
        return [255, 255, 127]

##############
# User input # Solution taken from: https://stackoverflow.com/questions/2408560/python-nonblocking-console-input
##############


old_settings = None


def set_up_terminal_for_key_check():
    global old_settings
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())


def reset_terminal():
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def check_for_key(target_key):
    if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
        user_key = sys.stdin.read(1)
        if user_key == target_key:
            return True
        else:
            return False
    else:
        return False


# THIS HASN'T BEEN IMPLEMENTED PROPERLY YET
def get_pressed_keys():
    pressed_keys = []
    if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
        while True:
            user_key = sys.stdin.read(1)
            print(user_key)
            print(int(user_key))
            if user_key != '':
                pressed_keys.append(user_key)
            else:
                break
    return pressed_keys


####################
# IMAGE PROCESSING #
####################


def add_random_image_noise(input_image):
    output_image = np.copy(input_image)
    rgb_noise = np.random.uniform(-0.2, 0.2, 3)
    for x in range(input_image.shape[1]):
        for y in range(input_image.shape[0]):
            for c in range(3):
                new_value = input_image[y, x, c] + rgb_noise[c]
                new_value += np.random.uniform(-0.1, 0.1)
                if new_value < 0:
                    new_value = 0
                if new_value > 1:
                    new_value = 1
                output_image[y, x, c] = new_value
    return output_image
