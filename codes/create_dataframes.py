import os 
import matplotlib.pyplot as plt
import pandas as pd   



activations = ['sigmoid','exponential','relu']

alphas = ['1.6','1.8','2.0','2.2','2.4',
            '2.6','2.8','3.0','3.2',
            '3.4','3.6','3.8','4.0',
            '4.2','4.4','4.6' 
            ]


# alphas = [1.0,1.2,1.4,1.6,1.8,2.0,2.2]

path_dict = {}
dict_df_jacc = {}
dict_df_ssim = {}
dict_df_jcc_3d = {}


for alpha in alphas :

    for activation in activations :
        path_dict[(str(alpha),str(activation))] = os.path.join('thresholded','thresholded_alpha_'+str(alpha), str(activation))
                   

        df_jacc = pd.read_csv(os.path.join(path_dict[(alpha,activation)],"result_jacc_mean"),
                    sep=',',
                    skiprows = 1 ,
                    names = ['seuil','mean']
                    )

        df_jacc['alpha']=[ str(alpha) for x in range(len(df_jacc)) ]
        df_jacc['activation']=[ str(activation) for x in range(len(df_jacc)) ]


        df_ssim = pd.read_csv(os.path.join(path_dict[(alpha,activation)],"result_ssim_mean"),
                    sep=',',
                    skiprows = 1 ,
                    names = ['seuil','mean']
                    )

        df_ssim['alpha']=[ str(alpha) for x in range(len(df_ssim)) ]
        df_ssim['activation']=[ str(activation) for x in range(len(df_ssim)) ]
                
        df_jcc3D = pd.read_csv(os.path.join(path_dict[(alpha,activation)],"result_jacc3D_mean"),
                    sep=',',
                    skiprows = 1 ,
                    names = ['seuil','mean']
                    )

        df_jcc3D['alpha']=[ str(alpha) for x in range(len(df_jcc3D)) ]
        df_jcc3D['activation']=[ str(activation) for x in range(len(df_jcc3D)) ]




        dict_df_jacc[(alpha, activation)] = np.mean(df_jacc[""])
        dict_df_ssim[(alpha, activation)] = df_ssim
        dict_df_jcc_3d[(alpha, activation)] = df_jcc3D





# keys = list(dict_df_jacc.keys() ) 


# merged = dict_df_jacc[keys[0]]

# for i in range(len(keys)-1):
#       merged = pd.concat([merged,dict_df_jacc[keys[i+1]]])
    


# print(merged.head())
