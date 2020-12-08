import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np
import os
import json
from PIL import Image  
import PIL  
from tqdm import tqdm
import cv2

for file in tqdm(os.listdir()):
    if file.endswith('.json'):
        f = open(file)
        im_data = json.load(f)
        img_extr = cv2.imread(im_data["imagePath"])
        img_extr=cv2.resize(img_extr,(800,615))
        h, w, c = img_extr.shape
        img = np.zeros((h + 4000, w + 4000, 3), dtype=np.uint8)
        img[2000:h+2000, 2000:w+2000,:] = img_extr
        x_min = 1000000
        x_max = 1
        y_min = 1000000
        y_max = 1
        for shape in im_data["shapes"]:
            if shape["label"] == "discLoc" or shape["label"] == "disc":
                for point in shape["points"]:
                    y_min = min(y_min, int(point[0]))
                    y_max = max(y_max, int(point[0]))
                    x_min = min(x_min, int(point[1]))
                    x_max = max(x_max, int(point[1]))
        
        height = (y_max - y_min) / 2
        width = (x_max - x_min) / 2
        c_x = (x_min + x_max) / 2 + 2000
        c_y = (y_min + y_max) / 2 + 2000
        height *= 1.8
        width *= 1.8

        img = img[int(c_x - width):int(c_x + width),int(c_y - height):int(c_y + height),:]


        
        try:
            
            im = Image.fromarray(img)
            plt.imshow(img)
            plt.show()
            #im = im.save(os.path.join("glaucoma", im_data["imagePath"])) 
        except:
            print("bad")
            print(im_data["imagePath"])
            print(int(c_x - width),int(c_x + width),int(c_y - height),int(c_y + height))

            plt.imshow(img)
            plt.show()
        
