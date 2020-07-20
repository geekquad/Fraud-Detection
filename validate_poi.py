
# coding: utf-8

# In[1]:


import pickle
import sys
from feature_format import featureFormat, targetFeatureSplit


# In[2]:


data_dict = pickle.load(open("C:/Users/Geekquad/ud120-projects/final_project/final_project_dataset_modified_unix.pkl", "rb"))


# In[3]:


features_list = ["poi", "salary"]


# In[5]:


data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)


# In[7]:


import sklearn
from sklearn.cross_validation import train_test_split
from sklearn import svm


# In[16]:


features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state=42)


# In[17]:


from sklearn.model_selection import GridSearchCV

parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
svr = svm.SVC()
clf = GridSearchCV(svr, parameters)
clf.fit(features_train,labels_train)
print(clf.score(features_test, labels_test))

