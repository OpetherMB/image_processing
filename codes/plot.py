import os 
import matplotlib.pyplot as plt
import pandas as pd   


alpha = 1.2

path_ = [
        os.path.join('thresholded/thresholded_alpha_'+str(alpha), 'sigmoid'),
        os.path.join( 'thresholded/thresholded_alpha_'+str(alpha),'exponential'),
        os.path.join('thresholded/thresholded_alpha_'+str(alpha),'relu')
        ]



df_sigmoid = pd.read_csv(os.path.join(path_[0],"result_jacc_mean") 
                            ,sep=',',
                            skiprows = 1 ,
                            names = ['seuil','mean'] )
df_exponential= pd.read_csv(os.path.join(path_[1],"result_jacc_mean") 
                      ,sep=',',skiprows = 1 , names = ['seuil','mean'] )


df_relu = pd.read_csv(os.path.join(path_[2],"result_jacc_mean") 
                      ,sep=',',skiprows = 1 , names = ['seuil','mean'] )

# print(df_sigmoid.head(10))
# print(df_exponential.head(10))
# print(df_relu.head(10))
# gca stands for 'get current axis'
ax = plt.gca()

df_sigmoid.plot(kind = 'line',x='seuil',y='mean',color = 'red' ,label = 'sigmoid', ax = ax )
df_exponential.plot(kind = 'line',x='seuil',y='mean',color = 'blue', label = 'exponential', ax=ax)
df_relu.plot(kind = 'line',x='seuil',y='mean',color = 'purple', label= 'relu', ax=ax )

plt.show()