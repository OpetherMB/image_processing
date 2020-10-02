from PIL import Image, ImageChops

import os
import imageio
from skimage.transform import resize
from PIL import Image
import math 
import pandas as pd
import scipy.spatial as ss

import warnings
from skimage.measure import compare_ssim
from skimage.transform import resize
from scipy.stats import wasserstein_distance
import imageio
import numpy as np
import cv2


from keras import backend as K
import tensorflow as tf
import re


'''
this programe calculate metrics jaccard and jacc 3d and SSIM and save them in a file for pair images 
and also calculates the mean of these metrics and stored in  thresholded/thresholded_alpha_'''alpha/relu'''

'''


alphas = [1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0,4.2,4.4,4.6]

structural_sim = []
jacard_3D_index = []



def comparison_images(array_a,array_b,racine=3,show_inter_union_img=True):
    
    intersection=np.minimum(array_a,array_b)
    union=np.maximum(array_a,array_b)

    if(show_inter_union_img):
        Image.fromarray(intersection).show()
        Image.fromarray(union).show()

    return intersection.sum()**(1./racine)/union.sum()**(1./racine)



def iou_coef(y_true, y_pred, smooth=1):
            
     intersection = np.logical_and(y_true, y_pred)
     union = np.logical_or(y_true, y_pred)
     iou_score = np.sum(intersection) / np.sum(union)
     
     return iou_score


for alpha in alphas  :


        path_ = [
                os.path.join('thresholded/thresholded_alpha_'+str(alpha), 'sigmoid'),
                os.path.join( 'thresholded/thresholded_alpha_'+str(alpha),'exponential'),
                os.path.join('thresholded/thresholded_alpha_'+str(alpha),'relu')
                ]


        # image_ids = []

        for i in range(3):
            
            with open(os.path.join(path_[i],"result_jacc_mean"),"w") as mean, open(os.path.join(path_[i],"result_ssim_mean"),"w") as mean_ssim, open(os.path.join(path_[i],"result_jacc3D_mean"),"w") as mean_jcc_3d :
                
                mean.write("seuil , mean\n")                
                mean_ssim.write("seuil , mean\n")
                mean_jcc_3d.write("seuil , mean\n")


                for thresh in [0, 25, 50, 75, 100, 128, 153, 178, 204, 230]:
                    results_iou  = []
                    images_id = os.listdir(path_[i])
                    
                    path = os.path.join(path_[i],str(thresh))

                    true_list=[x for x in os.listdir(path) if(x.split("_")[-1]=='true.png')]
                    pred_list=[x for x in os.listdir(path) if(x.split("_")[-1]=='pred.png')]

                    true_list = sorted(true_list)
                    pred_list = sorted(pred_list)

                    for true,pred in zip(true_list,pred_list):

                                img_path_true = os.path.join(path,true)
                                img_path_pred = os.path.join(path,pred)

                                img_true = Image.open(img_path_true)
                                y_true = np.array(img_true)
                                
                                img_pred = Image.open(img_path_pred)
                                y_pred = np.array(img_pred)

                                results =  iou_coef(y_true,y_pred)
                                id_name =  true.split("true.png")[0]

                                sim, diff = compare_ssim(y_true, y_pred, full=True)
                                jaccard_3d = comparison_images(y_true, y_pred , racine= 3 , show_inter_union_img=False)            

                                results_iou.append(results)
                                structural_sim.append(sim)
                                jacard_3D_index.append(jaccard_3d)

                                # image_ids.append(id_name)
                                # print(" results for " , id_name)
                                # print( " ~~~~~~ jaccard result is : ~~~~~~~~ " , results )

                                with open(os.path.join(path,"result_jaccard"),"a") as results_:
                                    results_.write("results for pair "+str(id_name)+" is : "+str(results)+"\n")
                                
                                with open(os.path.join(path,"results_SSIM"),"a") as file_ssim:
                                    file_ssim.write("results for pair  "+str(id_name)+" is : "+str(sim)+"\n")
                                
                                with open(os.path.join(path,"results_jaccard_3d"),"a") as file_3d:
                                    file_3d.write("results for pair  "+str(id_name)+" is : "+str(jaccard_3d)+"\n")     




                    print(" ")
                    print(" ~~~~~~~~~~ moyenne pr  jaccard ~~~~~~~~~~~ ", thresh," ~~~~~~~~~ le path : " , path_[i] )
                    print(" moyenne :", str(sum(results_iou)/len(results_iou)) )
                    print(" ~~~~~~~~~~ moyenne pr  SSIM ~~~~~~~~~~~ ", thresh," ~~~~~~~~~ le path : " , path_[i] )
                    print(" moyenne :", str(sum(structural_sim)/len(structural_sim)) )
                    print(" ~~~~~~~~~~ moyenne pr Jaccard 3d index  ~~~~~~~~~~~ ", thresh," ~~~~~~~~~ le path : " , path_[i] )
                    print(" moyenne :", str(sum(jacard_3D_index)/len(jacard_3D_index)) )
                    print("")



                    mean.write(str(thresh)+","+str(sum(results_iou)/len(results_iou))+"\n")            
                    mean_ssim.write(str(thresh)+","+str(sum(structural_sim)/len(structural_sim))+"\n")
                    mean_jcc_3d.write(str(thresh)+","+str(sum(jacard_3D_index)/len(jacard_3D_index))+"\n")

