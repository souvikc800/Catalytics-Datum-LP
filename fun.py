# -*- coding: utf-8 -*-
"""
Created on Wed May 18 13:48:20 2022

@author: Souvik Chakraborty
"""

import pandas as pd 
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
#Functions

#Extract the numerical features
def Diff(li1, li2):
    L3=list(set(li1) - set(li2)) + list(set(li2) - set(li1))
    return L3



#cleaning the imputation options
def CleanOption (a , df_no_num, df_no_num_copy):
    if a==1:
        df_no_num_cleaned_1 = df_no_num.dropna()
        return df_no_num_cleaned_1
        
    elif a==2:
        df_no_num_cleaned_2 = df_no_num.fillna(df_no_num_copy.mean())
        return df_no_num_cleaned_2
    
    elif a==3:
        
        df_no_num_cleaned_3 = df_no_num.fillna(df_no_num_copy.median())
        return df_no_num_cleaned_3
        


#scaling options
def ScalingOption(a , df_no_num_cleaned,df_no_num_cleaned_copy ):
    
    if a==1:
        #scaling the numerical features using MinMax Scaler
        from sklearn.preprocessing import MinMaxScaler 
        scaler1 = MinMaxScaler()
        scaler1.fit(df_no_num_cleaned) 
        df_no_num_cleaned_scaled = pd.DataFrame(scaler1.transform(df_no_num_cleaned),columns=df_no_num_cleaned.columns, index = df_no_num_cleaned.index)
        return df_no_num_cleaned_scaled
        
    elif a==2:
        #Z-score based scaling--give the choice
        from sklearn.preprocessing import StandardScaler
        scaler2=StandardScaler()
        scaler2.fit(df_no_num_cleaned)
        df_no_num_cleaned_scaled= pd.DataFrame(scaler2.transform(df_no_num_cleaned),columns=df_no_num_cleaned.columns, index = df_no_num_cleaned.index)
        return df_no_num_cleaned_scaled
    
#clustering with user-defined threshold  
def ClusterOption(a, X):
    wcss=[]
    for i in range(2,10):
        kmeans = KMeans(i)
        kmeans.fit(X)
        wcss_iter = kmeans.inertia_
        wcss.append(wcss_iter)
        
    for i in range(len(wcss)-1):
              if (abs((wcss[i]- wcss[i+1]) / wcss[i]) <= a):
                  return i+1
                  break
                  
              

    
    