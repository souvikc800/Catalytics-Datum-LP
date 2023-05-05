# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:32:15 2022

@author: Souvik Chakraborty

"""

## Customer Churn Prediction using K-Means Clustering

# Steps 

# 1.Data cleaning by removing categorical features

# 2.Handling missing values with drop & imputation options

# 3.Scaling the numerical features using z-scaler or minimaz scaling options

# 4.Feature selection using correlation analysis

# 5.Performing K-Means clustering using user defined threshold

# 6.Cluster profiling for the selected cluster with descriptive statistics


import pandas as pd 
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

#importing custom module with necessary functions
import fun



# #Functions

# def Diff(li1, li2):
#     L3=list(set(li1) - set(li2)) + list(set(li2) - set(li1))
#     return L3




# def CleanOption (a , df_no_num, df_no_num_copy):
#     if a==1:
#         df_no_num_cleaned_1 = df_no_num.dropna()
#         return df_no_num_cleaned_1
        
#     elif a==2:
#         df_no_num_cleaned_2 = df_no_num.fillna(df_no_num_copy.mean())
#         return df_no_num_cleaned_2
    
#     elif a==3:
        
#         df_no_num_cleaned_3 = df_no_num.fillna(df_no_num_copy.median())
#         return df_no_num_cleaned_3
        



# def ScalingOption(a , df_no_num_cleaned,df_no_num_cleaned_copy ):
    
#     if a==1:
#         #scaling the numerical features using MinMax Scaler
#         from sklearn.preprocessing import MinMaxScaler 
#         scaler1 = MinMaxScaler()
#         scaler1.fit(df_no_num_cleaned) 
#         df_no_num_cleaned_scaled = pd.DataFrame(scaler1.transform(df_no_num_cleaned),columns=df_no_num_cleaned.columns, index = df_no_num_cleaned.index)
#         return df_no_num_cleaned_scaled
        
#     elif a==2:
#         #Z-score based scaling--give the choice
#         from sklearn.preprocessing import StandardScaler
#         scaler2=StandardScaler()
#         scaler2.fit(df_no_num_cleaned)
#         df_no_num_cleaned_scaled= pd.DataFrame(scaler2.transform(df_no_num_cleaned),columns=df_no_num_cleaned.columns, index = df_no_num_cleaned.index)
#         return df_no_num_cleaned_scaled
    
    
    


#creating the churn_No Dataset

df=pd.read_excel("cell2celltrain.xlsx")

df['Rev_invalid'] = (df['MonthlyRevenue'].isna() | (df['MonthlyRevenue'] == 0) )

df['mins_invalid'] = (df['MonthlyMinutes'].isna()  | (df['MonthlyMinutes'] == 0))

df_1=df[~(df["Rev_invalid"] & df["mins_invalid"])]

#print(df_1.columns)
df_1=df_1.drop(["CustomerID"], axis=1)

#df_yes=df_1[df_1["Churn"]=="Yes"]

df_no=df_1[df_1["Churn"]=="No"]


#Logging-1
name="churn_no.xlsx"
df_no.to_excel(name)



#extracting numerical features

L2=list(df_no.select_dtypes(['object']).columns)
L1=df_no.columns


L4=fun.Diff(L1,L2)
df_no_num= df_no[L4]


#Logging-2
name="churn_no_num.xlsx"
df_no_num.to_excel(name)



#cleaning/ Imputing

# df_no_num_cleaned_1 = df_no_num.dropna()

# df_no_num_cleaned_2 = df_no_num.fillna(df_no_num.mean(), inplace=True)

# df_no_num_cleaned_3 = df_no_num.fillna(df_no_num.median(), inplace=True)

b=2
#calling clean options
df_no_num_cleaned = fun.CleanOption(b,df_no_num,df_no_num )
#df_no_num_cleaned.head()

#Logging-3
name="churn_no_num_cleaned.xlsx"
df_no_num_cleaned.to_excel(name)


#scaling the numerical features using MinMax Scaler

# from sklearn.preprocessing import MinMaxScaler 
# scaler1 = MinMaxScaler()
# scaler1.fit(df_no_num_cleaned_2) 
# df_no_num_cleaned_2_scaled = pd.DataFrame(scaler1.transform(df_no_num_cleaned_2),columns=df_no_num_cleaned_2.columns, index = df_no_num_cleaned_2.index)


# #Z-score based scaling--give the choice

# from sklearn.preprocessing import StandardScaler

# scaler2=StandardScaler()
# scaler2.fit(df_no_num_cleaned_2)
# df_no_num_cleaned_2_scaled_z= pd.DataFrame(scaler2.transform(df_no_num_cleaned_2),columns=df_no_num_cleaned_2.columns, index = df_no_num_cleaned_2.index)

#scaling

c=1

#calling scaling options
df_no_num_cleaned_scaled = fun.ScalingOption(c,df_no_num_cleaned,df_no_num_cleaned )

#Logging-4
name="churn_no_num_cleaned_scaled.xlsx"
df_no_num_cleaned_scaled.to_excel(name)



#starting correlation analysis

df_3=df_no_num_cleaned_scaled.corr()

#Numerical Features for MonthlyRevenue 
df_4=df_3.loc['MonthlyRevenue'][df_3.loc['MonthlyRevenue'] >0.35]
L5=list(df_4.index)


#Numerical Features for MonthlyMinutes 
df_5=df_3.loc['MonthlyMinutes'][df_3.loc['MonthlyMinutes'] >0.35]
L6=list(df_5.index)

#Taking the features common to both MonthlyRevenue & MonthlyMinutes
f=list(set(L5).intersection(L6))
print("Final Numerical Feature set for  no churn customers: ", f )






#K-Means Clustering
df_final=df_no_num_cleaned_scaled[f]
df_final.head()


X=np.array(df_final)
#print(X)


#WCSS method to find the optimal no of clusters

# wcss=[]
# for i in range(2,10):
#     kmeans = KMeans(i)
#     kmeans.fit(X)
#     wcss_iter = kmeans.inertia_
#     wcss.append(wcss_iter)


# number_clusters = range(2,10)
# plt.plot(number_clusters,wcss)
# plt.title('The Elbow title')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')

#Elbow point is 4.



t=0.05

#k is the number of clusters as per user selected threshold
k= fun.ClusterOption(t, X)
print("Number of clusters as per the threshold",k)

#calling clustering function
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=500, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X) #cluster label colm

#print(pred_y)

df_new=df_final
df_new["clusters"]=pred_y

#plotting the clusters
fig, ax = plt.subplots()
sc=ax.scatter(df_new['MonthlyRevenue'],df_new['MonthlyMinutes'],c=df_new['clusters'],cmap='rainbow',label=[0,1,2,3])
ax.legend(*sc.legend_elements(), title='clusters')
plt.xlabel('MonthlyRevenue')
plt.ylabel('MonthlyMinutes')


#print(df_new['clusters'])

#mapping the centroids
#print(kmeans.cluster_centers_)
#print((kmeans.cluster_centers_).size)
plt.scatter(df_new['MonthlyRevenue'], df_new['MonthlyMinutes'])
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=500, c='red')
plt.show()


#Evaluation
#Silhoutte Analysis -->  Coefficient should be close to 1 for a good clustering 

silhouette_vals = silhouette_samples(X, pred_y)

avg_score = np.mean(silhouette_vals)
print(avg_score)



#cluster profiling with individual viewing options
c=2
df_temp=df_new[df_new['clusters']==c]
plt.scatter(df_temp['MonthlyRevenue'], df_temp['MonthlyMinutes'])
plt.xlabel('MonthlyRevenue')
plt.ylabel('MonthlyMinutes')


print("No of customers in this cluster",df_temp.shape[0]) # no of customers in that cluster
print("Summary of Monthly  Revenue",df_temp["MonthlyRevenue"].describe())
print("Summary of Monthly usage in mins",df_temp["MonthlyMinutes"].describe())


fig, ax = plt.subplots(figsize =(5, 5))
ax.hist(df_temp['MonthlyRevenue'], bins = [0,0.1, 0.2,0.3,0.4])
plt.show

fig1, ax1 = plt.subplots(figsize =(5, 5))
ax1.hist(df_temp['MonthlyMinutes'], bins = [0,0.1, 0.2,0.3,0.4])
plt.show

#print(df_temp['MonthlyRevenue'])
