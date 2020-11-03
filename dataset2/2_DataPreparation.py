#!/usr/bin/env python
# coding: utf-8

# # 2. Data preparation
# 
# ---

# ### Setup

# In[6]:


import pandas as pd

original: pd.DataFrame = pd.read_csv('qsar_oral_toxicity_after_profiling.csv', sep=';')
bool_vars = original.select_dtypes(include='bool')


# ## 2.0. Data preparation
# 
# ---

# ### Missing Values Imputation
# 
# ---

# In our dataset there are no missing values and therefore this step is not appliable.

# ### Outliers Imputation
# ---

# In[7]:


#TODO


# ### Scaling
# 
# ---

# Since all variables are binary (0 or 1) they are already scaled. Therefore, no scaling is needed.

# In[8]:


original.to_csv('qsar_oral_toxicity_after_preparation.csv', sep=';', index=False)


# ### Summary
# 
# ---
# 
# ***Are all variables in the same scale? If not, how does scaling impact the results?***
# 
# Yes, they are all binary and, therefore, all in the same scale.

# ### Feature Selection
# 
# ---

# #### Unsupervised Selection

# By definition, unsupervised selection only aims for eliminating redundancies among the
# variables, getting the smallest set possible.

# In[11]:


from scipy.spatial.distance import pdist, squareform
import numpy as np


def distcorr(X, Y):
    """ Compute the distance correlation function
    
    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    0.762676242417
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X).astype('float32'))
    b = squareform(pdist(Y).astype('float32'))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    
    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor


# In[12]:


import seaborn as sns
import numpy as np

copy = original.copy(deep=True)

removed = []

for x in copy.columns:
    for y in copy.columns:
        if x == y:
            break
        corr = distcorr(original[x], original[y])
        if abs(corr) <= 0.1 and x not in removed and y not in removed:
            original = original.drop(x, axis=1)
            removed.append(x)
            break

#X = original.iloc[:,0:986]
#y = original.iloc[:,986].values
#full_data= X.copy()
#full_data['exp'] = y


# #### Supervised Selection

# In the context of supervised selection, the goal is to identify the most relevant variables
# in relation to the target variable, and so we need criteria able to relate each variable
# with the target one.
# We will use sequential backward selection to select our sets of variables.

# In[35]:


importances = full_data.drop('exp', axis=1).apply(lambda x: x.corr(full_data.exp))
print(importances)


# In[46]:


for i in range(0, len(importances)):
    if np.abs(importances[i]) > 0.15:
        print(importances.index[i])


# In[53]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd

X = pd.get_dummies(X)
y = pd.get_dummies(y)

for k in range(0, 987):
    #feature selection using chi2
    bestfeatures = SelectKBest(score_func=chi2, k=k)
    fit = bestfeatures.fit(X, y)
    #create df for scores
    dfscores = pd.DataFrame(fit.scores_)
    #create df for column names
    dfcolumns = pd.DataFrame(X.columns)

    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    #naming the dataframe columns
    featureScores.columns = ['Selected_columns','Score_chi2'] 
    #print 5 best features
    print(featureScores.nlargest(k,'Score_chi2'))


# In[ ]:




