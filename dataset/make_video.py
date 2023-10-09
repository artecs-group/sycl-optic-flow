import cv2
import numpy as np
import os
import sys
import glob

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Error: make_video.py <image folder path> <output video name>")
        exit(1)

    data_path: str = sys.argv[1]
    out_file: str = f"{sys.argv[2]}.avi"
    img_array = []
    for filename in sorted(os.listdir(data_path)):
        f_path: str = os.path.join(data_path, filename)
        print(f_path)
        img = cv2.imread(f_path)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)


    out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()