import cv2
import numpy as np
from time import time

cam = cv2.VideoCapture("./videos/video.mp4")

RESULT_IMAGE_HEIGHT = 3000
COLOR_BAR_WIDTH = 1

current_frame = 0
current_strip = 0
total_frames =  int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
number_of_color_strips = total_frames // 24
full_mean_image  = np.zeros([RESULT_IMAGE_HEIGHT, number_of_color_strips*COLOR_BAR_WIDTH,  3])
start_time = time()
while(current_frame < total_frames):
    return_code, frame = cam.read()
    if return_code:
        if current_frame % 24 == 0:
            mean_color = np.mean(frame, axis=(0,1))
            #Crate image with only mean color
            mean_image = np.ones([RESULT_IMAGE_HEIGHT, COLOR_BAR_WIDTH, 3]) * mean_color 
            full_mean_image[:, (current_strip*COLOR_BAR_WIDTH) : (current_strip*COLOR_BAR_WIDTH + COLOR_BAR_WIDTH), :] = mean_image
            current_strip += 1
        if current_frame % 1000 == 0:
            current_time = time()
            print(f'Progress = {(current_frame/total_frames):.4f}%, Elapsed time: {(current_time - start_time):.1f}s')
        current_frame += 1
    else:
        break

cv2.imwrite('results/result.jpg', full_mean_image)
cam.release()
cv2.destroyAllWindows()