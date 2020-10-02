import os,errno
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import copy

import numpy as np


'''
this programe create images thresholded using threshold list 
and store results thresholded/thresholded_alpha
'''


def makeDirectory(path):
    try:
        os.makedirs(path)
    except OSError as e:

        if e.errno != errno.EEXIST:
            raise





alphas = [1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0,4.2,4.4,4.6]


for alpha in alphas  :

    path = [os.path.join('Augmented/augmented_alpha_'+str(alpha), 'sigmoid'),
            os.path.join( 'Augmented/augmented_alpha_'+str(alpha),'exponential'),
            os.path.join('Augmented/augmented_alpha_'+str(alpha),'relu')
        ]



    pathToStore = [os.path.join('thresholded/thresholded_alpha_'+str(alpha), 'sigmoid'),
                    os.path.join( 'thresholded/thresholded_alpha_'+str(alpha),'exponential'),
                    os.path.join('thresholded/thresholded_alpha_'+str(alpha),'relu')
                    ]




    for j in range(3):
            
        print("processing data of ~~~ :", path[j] )

        id_ = os.listdir(path[j])

        for thresh in [0, 25, 50, 75, 100, 128, 153, 178, 204, 230]:
            
            for i in id_:

                    img_path = os.path.join(path[j], i)

                    img_y = Image.open(img_path)
                    img_y = np.array(img_y)
                    

                    img_y[img_y > thresh ] = 255  
                    img_y[img_y <= thresh ] = 0  

                    #save 
                    dir_percentage = os.path.join(pathToStore[j],str(thresh))
                    makeDirectory(dir_percentage)

                    cv2.imwrite(os.path.join(dir_percentage,i), img_y)



