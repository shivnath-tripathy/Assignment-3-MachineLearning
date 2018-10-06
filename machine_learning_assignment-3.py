
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
data_set = sm.datasets.fair.load_pandas().data

data_set.head()


# In[2]:


#Data Visualisation

get_ipython().magic('matplotlib inline')

data_set.educ.hist()
plt.title("Education")
plt.xlabel("Education Level")
plt.ylabel("Frequency")


# In[4]:


# Histogram of marriage rating
data_set.rate_marriage.hist()
plt.title('Histogram of Marriage Rating')
plt.xlabel('Marriage Rating')
plt.ylabel('Frequency')


# In[8]:


pd.crosstab(data_set.rate_marriage, data_set.affairs.astype(bool)).plot(kind= 'bar', color=['b','y'])
plt.title('Marriage Rating Distribution by Affair Status')
plt.xlabel('Marriage Rating')
plt.ylabel('Frequency')


# In[10]:


#Data Pre Processing

# add "affair" column: 1 represents having affairs, 0 represents not
data_set['affair'] = (data_set.affairs > 0).astype(int)
y, X = dmatrices('affair ~ rate_marriage + age + yrs_married + children + religious + educ + C(occupation) + C(occupation_husb)',
data_set, return_type="dataframe")


# fix column names of X
X = X.rename(columns = {'C(occupation)[T.2.0]':'occ_2',
'C(occupation)[T.3.0]':'occ_3',
'C(occupation)[T.4.0]':'occ_4',
'C(occupation)[T.5.0]':'occ_5',
'C(occupation)[T.6.0]':'occ_6',
'C(occupation_husb)[T.2.0]':'occ_husb_2',
'C(occupation_husb)[T.3.0]':'occ_husb_3',
'C(occupation_husb)[T.4.0]':'occ_husb_4',
'C(occupation_husb)[T.5.0]':'occ_husb_5',
'C(occupation_husb)[T.6.0]':'occ_husb_6'})

# flatten y into a 1-D array
y = np.ravel(y)


# In[11]:


X.head()


# In[12]:


y


# In[13]:


# instantiate a logistic regression model, and fit with X and y
model = LogisticRegression()
model = model.fit(X,y)
# check the accuracy on the training set
model.score(X, y)


# In[14]:


model.coef_


# In[15]:


dff  =  pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))),columns =['Feature', 'Coefficient'])


# In[16]:


dff


# In[17]:


# Train Data and Evaluate the Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state =0 )
model1 = LogisticRegression()
model1 = model1.fit(X_train, y_train)


# In[19]:


pred = model1.predict(X_test)

predComp = pd.DataFrame(list(zip(y_test, pred)), columns =['Actual', 'Predicted'])
predComp['Accuracy'] = (predComp.Actual == predComp.Predicted).astype(int)
print('Accuracy Percentage =', predComp['Accuracy'].sum()/predComp['Accuracy'].count())


# In[20]:


print(predComp.head())
print("Total Actually not having affair",predComp[predComp['Actual'] ==0].count() )
print("Total Actually having affair",predComp[predComp['Actual'] ==1].count() )


# In[21]:


print(metrics.accuracy_score(y_test, pred))


# In[22]:


# generate class probabilities
probs = model1.predict_proba(X_test)
probs


# In[23]:


#Confusion Matix
print(metrics.confusion_matrix(y_test, pred))

