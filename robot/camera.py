import pyrealsense2 as rs
import numpy as np
import sys
import cv2
import torch
import time
import traceback
import rospy

from common import config

class Camera:

    def __init__(self):
        self._pipeline = None
        self._cam_config = None
        self.image_num = 0

        self._initialise_camera()
        rospy.on_shutdown(self.shutdown)
    
    def _initialise_camera(self):
        # Get device product line for setting a supporting resolution
        self._pipeline = rs.pipeline()
        self._cam_config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self._pipeline)
        pipeline_profile = self._cam_config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        self._cam_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if device_product_line == 'L500':
            self._cam_config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            self._cam_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Uncoment is there is any issue with camera
        # print("reset start")
        # ctx = rs.context()
        # devices = ctx.query_devices()
        # for dev in devices:
        #     dev.hardware_reset()
        # print("reset done")
        
        # Start streaming
        self._pipeline.start(self._cam_config)
        print('Camera initialised.')
    
    def _capture_image(self):
        try:
            frames = self._pipeline.wait_for_frames()
            
            color_frame = frames.get_color_frame()   
            return color_frame
        except Exception:
            print(traceback.format_exc())

    def capture_cv_image(self, resize_image=False, show_image=False, show_big_image=False):
        try:
            color_frame = self._capture_image()
            # Convert images to numpy arrays
            bgr_image = np.asanyarray(color_frame.get_data())

            if bgr_image is None:
                sys.exit("Could not read the image.")

            cv2.imwrite('../Results/Trajectory_Images/image_' + str(self.image_num) + '.png', bgr_image)
            self.image_num += 1
            if resize_image:
                resized_image = cv2.resize(bgr_image, dsize=(config.RESIZED_IMAGE_SIZE, config.RESIZED_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
            else:
                resized_image = bgr_image
            if show_image:
                if show_big_image:
                    big_size = (1024, 1024)
                    big_image = cv2.resize(bgr_image, big_size)
                    cv2.imshow('Big Live Image', big_image)
                else:
                    cv2.imshow('Live Image', bgr_image)
                cv2.waitKey(1)
            return resized_image
        except Exception:
            print(traceback.format_exc())   

    # Capture an RGB image in PyTorch format: colour channel is first axis, data type is float32, range is 0 to 1
    def capture_torch_image(self, resize_image=False, show_image=False, show_big_image=False):
        cv_image = self.capture_cv_image(resize_image, show_image, show_big_image)
        torch_image = torch.tensor(np.moveaxis(cv_image, 2, 0).astype(np.float32))
        torch_image /= 255.0
        return torch_image

    def show_live_image(self):
        image = self.capture_cv_image(resize_image=False)
        cv2.imshow('Live Image', image)
        cv2.waitKey(1)

    def show_big_live_image(self):
        image = self.capture_cv_image(resize_image=False)
        big_size = (1024, 1024)
        big_image = cv2.resize(image, big_size, cv2.INTER_NEAREST)
        cv2.imshow('Big Image Nearest', big_image)
        cv2.waitKey(1)

    def test_image_capture(self):
        if 1:
            while True:
                self.show_live_image()
        if 0:
            images = []
            for i in range(10):
                t1 = time.time()
                image = self.capture_cv_image()
                t2 = time.time()
                print('frame rate = ' + str(1.0/(t2-t1)))
                images.append(image)
            for i in range(10):
                cv2.imwrite('image_' + str(i) + '.bmp', images[i])

    def shutdown(self):
        print('Shutting down camera ...')
        self._pipeline.stop()
        del self._pipeline
        del self._cam_config
        print('\tCamera shutdown')
        