#!/usr/bin/env python
# coding: utf-8

# In[62]:


import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

from sklearn.datasets import load_digits


# In[5]:


handwrittendigitdata = load_digits()


# # Kernel Density Estimation

# In[69]:


bandwidthmin= -1.0
bandwidthmax= 1.0
n = 20
kde = []
comp = [15,30,41]
pca =[]
for c in comp:
    pca_ = PCA(n_components=c)
    reduceddata = pca_.fit_transform(handwrittendigitdata.data)
    pca.append(pca_)
    gridsearch = GridSearchCV(KernelDensity(),{'bandwidth': np.linspace(bandwidthmin, bandwidthmax, n)},cv = 3)
    gridsearch.fit(reduceddata)
    print(gridsearch.best_params_['bandwidth']) 
    kde_ = gridsearch.best_estimator_
    kde.append(kde_)


# # Gaussian Mixture Model based Density Estimation:

# In[75]:


mincomp = []
for n in comp:  
    pca_ = PCA(n_components=n)
    reduceddata = pca_.fit_transform(handwrittendigitdata.data)
    n_components = np.arange(1, n)
    gaussianmodels = [GaussianMixture(n, covariance_type='full', random_state=0).fit(reduceddata)
          for n in n_components]
    plt.plot(n_components, [m.bic(reduceddata) for m in gaussianmodels], label='BIC')
    print(n)
    mincomp.append(np.argmin([m.bic(reduceddata) for m in gaussianmodels]))


# In[76]:


print(mincomp)


# # Part-3:

# In[78]:


for i in range(3):
    new_data = kde[i].sample(44, random_state=0)
    new_data = pca[i].inverse_transform(new_data)

    new_data = new_data.reshape((4, 11, -1))
    real_data = handwrittendigitdata.data[:44].reshape((4, 11, -1))
    
    fig, ax = plt.subplots(9, 11, subplot_kw=dict(xticks=[], yticks=[]))
    for j in range(11):
        ax[4, j].set_visible(False)
        for i in range(4):
            im = ax[i, j].imshow(real_data[i, j].reshape((8, 8)),
                                 cmap=plt.cm.binary, interpolation='nearest')
            im.set_clim(0, 16)
            im = ax[i + 5, j].imshow(new_data[i, j].reshape((8, 8)),
                                     cmap=plt.cm.binary, interpolation='nearest')
            im.set_clim(0, 16)

    ax[0, 5].set_title('Selection from the input data')
    ax[5, 5].set_title('"New" digits drawn from the kernel density model')

    plt.show()


# In[84]:


for i in range(3):
    data = pca[i].fit_transform(handwrittendigitdata.data)
    gmm = GaussianMixture(n_components=mincomp[i],covariance_type="full")
    gmm.fit(data)
    new_data = gmm.sample(44)
    new_data = new_data[0]
    
    
    new_data = pca[i].inverse_transform(new_data)

    new_data = new_data.reshape((4, 11, -1))
    real_data = handwrittendigitdata.data[:44].reshape((4, 11, -1))
    
    fig, ax = plt.subplots(9, 11, subplot_kw=dict(xticks=[], yticks=[]))
    for j in range(11):
        ax[4, j].set_visible(False)
        for i in range(4):
            im = ax[i, j].imshow(real_data[i, j].reshape((8, 8)),
                                 cmap=plt.cm.binary, interpolation='nearest')
            im.set_clim(0, 16)
            im = ax[i + 5, j].imshow(new_data[i, j].reshape((8, 8)),
                                     cmap=plt.cm.binary, interpolation='nearest')
            im.set_clim(0, 16)

    ax[0, 5].set_title('Selection from the input data')
    ax[5, 5].set_title('"New" digits drawn from the kernel density model')

    plt.show()


# In[ ]:




