from robot.cam_controller import Camera
import asyncio

camera = Camera()

camera.test_image_capture()
#asyncio.run(camera.capture_cv_image(resize_image=False, show_image=True, show_big_image=False))

#camera.show_live_image()
camera.shutdown()