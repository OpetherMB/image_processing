import os 
import matplotlib.pyplot as plt
import pandas as pd   


''' plotting results '''


# path_ = ['thresholded/thresholded_alpha_1.2/sigmoid',
#         'thresholded/thresholded_alpha_1.2/exponential',
#         'thresholded/thresholded_alpha_1.2/relu'
#         ]

# pathTocmp = ['thresholded/thresholded_alpha_1.0/sigmoid',
#         'thresholded/thresholded_alpha_1.0/exponential',
#         'thresholded/thresholded_alpha_1.0/relu'
#         ]


activations = ['sigmoid','exponential','relu']
alphas = ['1.6','1.8','2.0','2.2','2.4',
            '2.6','2.8','3.0','3.2',
            '3.4','3.6','3.8','4.0',
            '4.2','4.4','4.6' 
            ]

alphas = [1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0,4.2,4.4,4.6]
# alphas = [1.0,1.2,1.4,1.6,1.8,2.0,2.2]

path_dict = {}
dict_df = {}


for alpha in alphas :
    for activation in activations :
        path_dict[(str(alpha),str(activation))] = os.path.join('thresholded','thresholded_alpha_'+str(alpha), str(activation))
                   

    for activation in activations :

        df = pd.read_csv(os.path.join(path_dict[(alpha,activation)],"result_jacc_mean"),
                        sep=',',
                        skiprows = 1 ,
                        names = ['seuil','mean']
                        )

        dict_df[(alpha, activation)] = df 



fig = plt.figure()

     

for i,key in enumerate(dict_df.keys()):

    if key[1] =='sigmoid': 
        if key[0] == '1.0' :
            plt.plot(dict_df[key]['seuil'], dict_df[key]['mean'] , label = str(key) , color ='red',  linestyle='dashed' )
        else : 
            plt.plot(dict_df[key]['seuil'], dict_df[key]['mean'] , label = str(key),  linestyle='dashed' )
    
    if key[1] =='relu': 
            if key[0] == '1.0' :
                plt.plot(dict_df[key]['seuil'], dict_df[key]['mean'] , label = str(key) , color ='green', linestyle='dotted' )
            else : 
                plt.plot(dict_df[key]['seuil'], dict_df[key]['mean'] , label = str(key), linestyle='dotted'   )
   
    if key[1] =='exponential': 
        if key[0] == '1.0' :
            plt.plot(dict_df[key]['seuil'], dict_df[key]['mean'] , label = str(key) , color ='black' , linestyle='dashdot' )
        else : 
            plt.plot(dict_df[key]['seuil'], dict_df[key]['mean'] , label = str(key), linestyle='dashdot'   )


plt.xlabel('seuil')
plt.ylabel('mean')
plt.legend()    
plt.show()    