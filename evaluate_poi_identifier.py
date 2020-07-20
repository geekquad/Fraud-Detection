
# coding: utf-8

# In[1]:


import pickle
import sys
from feature_format import featureFormat, targetFeatureSplit


# In[2]:


data_dict = pickle.load(open("C:/Users/Geekquad/ud120-projects/final_project/final_project_dataset_modified_unix.pkl", "rb"))


# In[3]:


features_list = ['poi', 'salary']
data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)


# In[10]:


import sklearn
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm


# In[6]:


features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state=42)


# In[7]:


clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
print("Accuracy:", clf.score(features_test, labels_test))
print(clf.predict(features_test))


# In[12]:


print('np.array(labels_test):')
print(np.array(labels_test))


# In[16]:


print('POIs predict:', clf.predict(features_test))
print('Number of POIs predict:', len([e for e in labels_test if e == 1.0]))


# In[17]:


print("Number of tests:", len(labels_test))


# In[21]:


from sklearn.metrics import *
print("precision:", precision_score(labels_test, clf.predict(features_test)))
print("recall:", recall_score(labels_test, clf.predict(features_test)))

