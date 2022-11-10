import numpy as np
import cv2 as cv
import glob
import sys
import pyrealsense2 as rs


### Capture images
def capture_images():
    image_dir='images/sample/'

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
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

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    print("reset start")
    ctx = rs.context()
    devices = ctx.query_devices()
    for dev in devices:
        dev.hardware_reset()
    print("reset done")
    # Start streaming
    pipeline.start(config)
    filename=11
    print('To save the image press s')
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert images to numpy arrays
            img = np.asanyarray(color_frame.get_data())

            if img is None:
                sys.exit("Could not read the image.")

            cv.namedWindow('RealSense', cv.WINDOW_AUTOSIZE)
            cv.imshow("Display window", img)
            k = cv.waitKey(1)

            if k == ord("s"):
                cv.imwrite(image_dir+str(filename)+".png", img)
                print(str(filename)+".png saved.")
                filename += 1

    finally:

        # Stop streaming
        pipeline.stop()


### Calibrate camera
def calibrate_camera():
    image_dir='images/cam_calibration/'
    save_dir='camera_data/'

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32) # chessboard 6x7
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob(image_dir + '*.png')
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (7,6), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (7,6), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(500)
    cv.destroyAllWindows()

    print('Starting Calibration')
    ret, cam_mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print('Camera Matrix')
    print(cam_mtx)
    np.save(save_dir+'cam_mat.npy', cam_mtx)

    print('Distortion Coeff')
    print(dist)
    np.save(save_dir+'dist.npy', dist)

    print('r_vecs')
    print(rvecs)
    #np.save(save_dir+'cam_mat.npy', cam_mtx)

    print('t_vecs')
    print(tvecs)
    #np.save(save_dir+'cam_mat.npy', cam_mtx)