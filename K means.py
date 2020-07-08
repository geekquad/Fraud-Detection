
# coding: utf-8

# In[22]:


import pickle
import numpy
import sys
import matplotlib.pyplot as plt
sys.path.append(".../tools/")
from feature_format import featureFormat, targetFeatureSplit


# In[23]:


def Draw(pred, features, poi, mark_poi=False, name='image.png', f1_name='feature 1', f2_name ='feature 2'):
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color= colors[pred[ii]])
    
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
                
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()


# In[24]:


data_dict = pickle.load(open("C:/Users/Geekquad/final_project_dataset_modified_unix.pkl", "rb"))
data_dict.pop("TOTAL", 0)


# In[25]:


feature_1 = 'salary'
feature_2 = "exercised_stock_options"
poi = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list)
poi, finance_features = targetFeatureSplit(data)


# In[26]:


for f1, f2 in finance_features:
    plt.scatter( f1, f2 )
plt.show()


# ## Clustering using K-means:

# In[27]:


from sklearn.cluster import KMeans
features_list = ["poi", feature_1, feature_2]
data2 = featureFormat(data_dict, features_list)
poi, finance_features = targetFeatureSplit(data2)
clf = KMeans(n_clusters=2)
pred = clf.fit_predict(finance_features)
Draw(pred, finance_features, poi, name="clusters_before_scaling.pdf", f1_name = feature_1, f2_name=feature_2)

