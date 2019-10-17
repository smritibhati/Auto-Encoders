#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
df = pd.read_csv('data.csv')
import csv
import copy
import matplotlib.pyplot as plt
import random as rd
import csv
import copy
import math
from collections import Counter

import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering

from sklearn.preprocessing import LabelEncoder


# In[2]:


# data,testdata= np.split(df,[int(0.80*len(df))])
X = df[df.columns[0:-1]]
Y = df[df.columns[-1]]


# In[3]:


means = np.mean(X, axis = 0)
stdDev = np.std(X, axis = 0)
X = (X - means) / stdDev


# In[4]:


def sigmoid(x):
    return 1.0/(1.0+ np.exp(-x))

def sigmoidderivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


# In[5]:


class NeuralNetwork:
    def __init__(self, X, Y, hiddeninfo):
        self.n = len(hiddeninfo)
        self.inputlayersize  = X.shape[1]
        self.outputlayersize = X.shape[1]
        self.allweights = [1] *(self.n + 1)
        self.allactivations = [1] *(self.n+1 )
        self.allz = [1]*(self.n +1)
        
        self.allweights[0] =  np.random.rand(self.inputlayersize,hiddeninfo[0])
        
        for i in range (1,self.n+1):
            if i==self.n:
                weights= np.random.rand(hiddeninfo[self.n-1],self.outputlayersize)
            else:
                weights = np.random.rand(hiddeninfo[i-1],hiddeninfo[i])
            self.allweights[i]=weights 
    
    def getreduceddimensions(self):
        return self.allactivations[1]
    
    def forwardprop(self,X):       
        output = []
        self.allz[0] = np.dot(X,self.allweights[0])
        self.allactivations[0] = sigmoid(self.allz[0])
        
        for i in range(1, len(self.allweights)):
            self.allz[i] = np.dot(self.allactivations[i-1],self.allweights[i])
            self.allactivations[i] = sigmoid(self.allz[i])

        return self.allactivations[-1]

    def backprop(self,X,Y):

        deltas = [np.float128(1.0)] * len(self.allweights)
        dweights = [np.float128(1.0)] *len(self.allweights)
        
        deltas[-1] = - (Y - X) * sigmoidderivative(self.allactivations[-1]);
        dweights[-1] = np.dot(self.allactivations[-2].T, deltas[-1])
        
#         print(self.allweights[0][0])
        i = len(self.allweights)-2
        while i>0:
            deltas[i] = np.dot(deltas[i+1],self.allweights[i+1].T)*sigmoidderivative(self.allactivations[i])
            dweights[i] = np.dot(self.allactivations[i-1].T, deltas[i])
            i-=1
        
        deltas[0] =  np.dot(deltas[1],self.allweights[1].T)*sigmoidderivative(self.allactivations[0])
        dweights[0] = np.dot(X.T, deltas[0])

        for i in range(len(self.allweights)):
            self.allweights[i] = self.allweights[i] - .0001 * dweights[i]
        
#         print (self.op)
#         print(dweights)


# In[6]:


def error(y, op):
#     print("Actual: ", y)
#     print("Derived: ", op)
    return np.mean(np.mean((y-op)**2))


# In[7]:


nn = NeuralNetwork(X,Y,[20,14,20])
ferrors=[]
errors=[]
for i in range(10):
    s = 0
    for i in range(12499):
        op = nn.forwardprop(X.iloc[s:s+2])
        outY= X.iloc[s:s+2]
#         print(error(outY,op))
        errors.append(error(outY,op))
        nn.backprop(op,outY)
        s+=2
    
    opall = nn.forwardprop(X)
    finalerror = error(X, opall)
    ferrors.append(finalerror)
    print(finalerror)


# In[8]:


plt.plot(ferrors)
plt.show()


# # K Means

# In[9]:


data = nn.getreduceddimensions()
reduceddf = pd.DataFrame(data=data[0:,0:])


# In[10]:


inputdata = reduceddf
mean=np.mean(inputdata)
standarddev=np.std(inputdata)
for i in range(len(inputdata.columns)):
    inputdata[inputdata.columns[i]]= [(1.0 * (colum-mean[i]))/standarddev[i] for colum in inputdata[inputdata.columns[i]]]
    


# In[11]:


inputdata = np.array(inputdata)
numoftrainingeg = inputdata.shape[0] 
numoffeatures =inputdata.shape[1] #number of features. Here n=2
k=5
random=rd.randint(0,numoftrainingeg-1)
centers= np.array(inputdata[random])

for i in range(4):
    random=rd.randint(0,numoftrainingeg-1)
    centers = np.vstack((centers,inputdata[random]))


# In[12]:


error = 99999
distances = np.zeros((numoftrainingeg,k))

while error >= 0.000001:
    for i in range(k):
        distances[:,i] = np.linalg.norm(inputdata - centers[i], axis=1)
    
    clusters = np.argmin(distances, axis = 1)
#     print(clusters)
    
    centersold = deepcopy(centers)

    for i in range(k):
        if math.isnan( np.mean(inputdata[clusters == i], axis=0)[0]):
            continue
        centers[i] = np.mean(inputdata[clusters == i], axis=0)
    
    error = np.linalg.norm(centers - centersold)
    print(error)


# In[13]:


def pre_dict(clusters, inputdata):
    predicted={}
    for i in range(len(clusters)):
        cl = clusters[i]
        if cl in predicted:
            predicted[cl].append(inputdata[i])
        else:
            predicted[cl]=[]
            predicted[cl].append(inputdata[i])
    
    purity=[]
    for cl in predicted:
        keys=[]
    
        for onelist in predicted[cl]:
            keys.append(onelist[14])
    
        c = Counter(keys)
        print(len(keys))
        value, count = c.most_common() [0]
        purity.append((value, (count/len(keys))))
    return purity


# In[14]:


reduceddf = pd.DataFrame(inputdata)
reduceddf[29] = np.array(df[df.columns[-1]])

inputdata = np.array(reduceddf)
purity = pre_dict(clusters,inputdata)


# In[15]:


averagekmeans=0
for i in range(len(purity)):
    print(purity[i][0], purity[i][1])
    averagekmeans+=purity[i][1]
averagekmeans/=len(purity)


# # GMM

# In[16]:


data = nn.getreduceddimensions()
reduceddf = pd.DataFrame(data=data[0:,0:])


# In[17]:


inputdata = np.array(reduceddf)

GMM = GaussianMixture(n_components=5).fit(inputdata)
print('Successful!',GMM.converged_)

df_1 = pd.DataFrame(inputdata)
predictionvalues = GMM.predict(inputdata)

df_1[29] = np.array(df[df.columns[-1]])

inputdata = np.array(df_1)
purity = pre_dict(predictionvalues, inputdata)


# In[18]:


averagegmm=0
for i in range(len(purity)):
    print(purity[i][0], purity[i][1])
    averagegmm+=purity[i][1]
averagegmm/=len(purity)


# # Heirarchical Clustering

# In[19]:


data = nn.getreduceddimensions()
reduceddf = pd.DataFrame(data=data[0:,0:])
inputdata = np.array(reduceddf)
hierClusters = AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto', connectivity=None, linkage='single', memory=None, n_clusters=5, pooling_func='deprecated') 
hierClusters.fit(inputdata)
hierClusters.labels_
df_1 = pd.DataFrame(inputdata)
df_1[29] = np.array(df[df.columns[-1]])
inputdata = np.array(df_1)
averageh=0
purity = pre_dict(hierClusters.labels_, inputdata)
for i in range(len(purity)):
    print(purity[i][0], purity[i][1])
    averageh+=purity[i][1]
averageh/=len(purity)


# In[20]:


labels = 'K-Means', 'GMM', 'Heirarchical'
sizes = [averagekmeans, averagegmm,averageh]
colors = ['gold', 'yellowgreen', 'lightblue']
explode = (0.1, 0.1,0.1)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()


# In[ ]:




