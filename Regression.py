
# coding: utf-8

# In[10]:


import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
import pickle

dictionary = pickle.load(open("C:/Users/Geekquad/ud120-projects/final_project/final_project_dataset_modified_unix.pkl", "rb"))


# In[11]:


features_list = ["bonus", "salary"]
data = featureFormat( dictionary, features_list, remove_any_zeroes=True)
target, features = targetFeatureSplit( data )


# In[12]:


from sklearn.cross_validation import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "b"


# In[15]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(feature_train, target_train)
print('coef', reg.coef_)
print('intercept', reg.intercept_)


# In[25]:


import matplotlib.pyplot as plt
for feature, target in zip(feature_test, target_test):
    plt.scatter(feature, target, color = test_color)
    
for feature, target in zip(feature_train, target_train):
    plt.scatter(feature, target, color= train_color)
    
plt.scatter(feature_test[0], target_test[0], color=test_color, label='test')
plt.scatter(feature_test[0], target_test[0], color=train_color, label='train')
plt.plot(feature_test, reg.predict(feature_test))
plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()

