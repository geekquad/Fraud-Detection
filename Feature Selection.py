
# coding: utf-8

# In[31]:


import pickle
import numpy
numpy.random.seed(42)


# In[32]:


word_file = "C:/Users/Geekquad/ud120-projects/feature_selection/word_data_modified_unix.pkl"
author_file = "C:/Users/Geekquad/ud120-projects/feature_selection/email_authors_modified_unix.pkl"
word_data = pickle.load(open(word_file, "rb"))
author_data = pickle.load(open(author_file, "rb"))


# In[33]:


import sklearn
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(word_data, author_data, test_size=0.1, random_state=42)


# In[34]:


from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words="english")
features_train = vect.fit_transform(features_train)
features_test = vect.transform(features_test).toarray()


# In[37]:


#### training only on 150 data points to put myself into overfit regime
features_train = features_train[:150].toarray()
labels_train = labels_train[:150]


# In[38]:


print('number of training points: ', len(features_train))


# In[45]:


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
