
# coding: utf-8

# In[51]:


import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
from outlier_cleaner import outlierCleaner


# In[52]:


ages = pickle.load( open("practice_outliers_ages_modified_unix.pkl", "rb") )
net_worths = pickle.load( open("practice_outliers_net_worths_modified_unix.pkl", "rb") )


# In[53]:


ages = np.reshape(np.array(ages), (len(ages), 1))
new_worths = np.reshape(np.array(net_worths), (len(net_worths), 1))

from sklearn.cross_validation import train_test_split
ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths, test_size=0.1, random_state = 42)


# In[54]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(ages_train, net_worths_train)
reg.predict(ages_test)
print('coef:', reg.coef_)
print('intercept:', reg.intercept_)
print('scores:', reg.score(ages_test, net_worths_test))


# In[55]:


plt.scatter(ages_train, net_worths_train, color='blue')
plt.scatter(ages_test, net_worths_test, color='red')
plt.plot(ages_train, reg.predict(ages_train), color='green')
plt.xlabel('Ages')
plt.ylabel('Net Worth')
plt.show()


# ## Removing the most outlier points:

# In[56]:


cleaned_data = []
try:
    predictions = reg.predict(ages_train)
    cleaned_data = outlierCleaner( predictions, ages_train, net_worths_train )
except NameError:
    print("your regression object doesn't exist, or isn't name reg")
    print("can't make predictions to use in identifying outliers")


# In[57]:


if len(cleaned_data) > 0:
    ages, net_worths, errors = zip(*cleaned_data)
    ages= np.reshape(np.array(ages), (len(ages), 1))
    net_worths = np.reshape( np.array(net_worths), (len(net_worths), 1))


# In[58]:


try:
    reg.fit(ages, net_worths)
    plt.plot(ages, reg.predict(ages), color="blue")
except NameError:
    print("you don't seem to have regression imported/created")
    print("or else your regression object isn't named reg")
    print("either way, only draw the scatter plot of the cleaned data")
    plt.scatter(ages, net_worths)
    plt.xlabel("ages")
    plt.ylabel("net worths")
    plt.show()

    print('coef-after cleaned:', reg.coef_)
    print('intercept-after cleaned:', reg.intercept_)
    print('score-after cleaned:', reg.score(ages_test, net_worths_test))

else:
    print("outlierCleaner() is returning an empty list, no refitting to be done")

