
# coding: utf-8

# In[20]:


import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
from outlier_cleaner import outlierCleaner


# In[22]:


ages = pickle.load( open("practice_outliers_ages_modified_unix.pkl", "rb") )
net_worths = pickle.load( open("practice_outliers_net_worths_modified_unix.pkl", "rb") )


# In[31]:


ages = np.reshape(np.array(ages), (len(ages), 1))
new_worths = np.reshape(np.array(net_worths), (len(net_worths), 1))

from sklearn.cross_validation import train_test_split
ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths, test_size=0.1, random_state = 42)


# In[37]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(ages_train, net_worths_train)
reg.predict(ages_test)
print('coef:', reg.coef_)
print('intercept:', reg.intercept_)
print('scores:', reg.score(ages_test, net_worths_test))


# In[39]:


plt.scatter(ages_train, net_worths_train, color='blue')
plt.scatter(ages_test, net_worths_test, color='red')
plt.plot(ages_train, reg.predict(ages_train), color='green')
plt.xlabel('Ages')
plt.ylabel('Net Worth')
plt.show()

