
# coding: utf-8

# In[83]:


import pickle
import numpy
numpy.random.seed(42)


# In[84]:


word_file = "C:/Users/Geekquad/ud120-projects/feature_selection/word_data_modified_unix.pkl"
author_file = "C:/Users/Geekquad/ud120-projects/feature_selection/email_authors_modified_unix.pkl"
word_data = pickle.load(open(word_file, "rb"))
author_data = pickle.load(open(author_file, "rb"))


# In[85]:


import sklearn
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(word_data, author_data, test_size=0.1, random_state=42)


# In[86]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words="english")
features_train = vectorizer.fit_transform(features_train)
features_test = vectorizer.transform(features_test).toarray()


# In[87]:


#### training only on 150 data points to put myself into overfit regime
features_train = features_train[:150].toarray()
labels_train = labels_train[:150]


# In[88]:


print('number of training points: ', len(features_train))


# In[89]:


"""overfitting the Decision Tree and cehcking the accuracy"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
y_pred = clf.predict(features_test)


print(confusion_matrix(labels_test, y_pred))
print(classification_report(labels_test, y_pred))
print(accuracy_score(labels_test, y_pred))


# Yes, it has an accuracy much higher than it should be. 
# Hence, finding the most important features.

# In[90]:


# identifying the most important features:
import numpy as np
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature Ranking")
for i in range(10):
    print("{} feature no.{} ({})".format(i+1,indices[i],importances[indices[i]]))


# In[91]:


vect.get_feature_names()[2802]


# This word seems like an outlier in a certain sense, so let’s remove it and refit. 

# In[99]:


word_file = "C:/Users/Geekquad/ud120-projects/feature_selection/word_data_overfit_modified_unix.pkl"
author_file = "C:/Users/Geekquad/ud120-projects/feature_selection/email_authors_overfit_modified_unix.pkl"
word_data = pickle.load(open(word_file, "rb"))
author_data = pickle.load(open(author_file, "rb"))


# In[100]:


import sklearn
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(word_data, author_data, test_size=0.1, random_state=42)


# In[101]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words="english")
features_train = vectorizer.fit_transform(features_train)
features_test = vectorizer.transform(features_test).toarray()


# In[102]:


#### training only on 150 data points to put myself into overfit regime
features_train = features_train[:150].toarray()
labels_train = labels_train[:150]


# In[103]:


"""overfitting the Decision Tree and cehcking the accuracy"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
y_pred = clf.predict(features_test)


print(confusion_matrix(labels_test, y_pred))
print(classification_report(labels_test, y_pred))
print(accuracy_score(labels_test, y_pred))


# In[104]:


# identifying the most important features:
import numpy as np
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature Ranking")
for i in range(10):
    print("{} feature no.{} ({})".format(i+1,indices[i],importances[indices[i]]))


# In[105]:


vectorizer.get_feature_names()[33604]

