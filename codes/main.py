import os,errno
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import copy
import numpy as np


'''
this program creates new images based on a scalar alpha
and store results seperatly in directory augmented/augmented_alpha_'''alpha'''/'''activation'''

you can change alphas list 

'''

def makeDirectory(path):
    try:
        os.makedirs(path)
    except OSError as e:

        if e.errno != errno.EEXIST:
            raise


path = ['poisson_40_dataset_full_centered_scale_autoscale_16_sigmoid/predit_grayscale'
        ,'poisson_40_dataset_full_centered_scale_autoscale_18_exponential/predit_grayscale'
        ,'poisson_40_dataset_full_centered_scale_autoscale_18_relu/predit_grayscale'
       ]


alphas = [1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0,4.2,4.4,4.6]

for alpha in alphas  : 

        
        pathToStore = [os.path.join('Augmented','augmented_alpha_'+str(alpha), 'sigmoid'),
                        os.path.join('Augmented', 'augmented_alpha_'+str(alpha),'exponential'),
                        os.path.join('Augmented','augmented_alpha_'+str(alpha),'relu')
                        ]

        for j in range(3):

            print("processing data of ~~~ :", path[j] )

            id_ = os.listdir(path[j])

            for i in id_:
            
                        img_path = os.path.join(path[j], i)
                        
                        id_name =  i.split("_")[6]
                        img_y = Image.open(img_path)

                        if id_name == 'pred.png' :                         
                            img_y = np.array(img_y) * alpha
                            img_y[img_y>255]= 255
                        
                        image = np.array(img_y)


                        #save 

                        makeDirectory(pathToStore[j])
                        cv2.imwrite(os.path.join(pathToStore[j],i), image )
                

