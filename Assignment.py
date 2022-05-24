#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os  #operating system
import logging   #create log to record results


#The following imported packages are data analysis packages
import pandas as pd
import numpy as np 
#Only importing the ols function from the statsmodels package
#Alternatively, we could do import statsmodels as sm then run sm.ols()
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt


# In[2]:


#Create log file
with open( "assignment.log", "w" ):
               pass
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


logging.basicConfig(filename="assignment.log", level=logging.INFO,
                   format='%(asctime)s:%(message)s')


# In[3]:


#load data
location = pd.read_excel("data_location.xlsx")
price = pd.read_excel("data_price.xlsx")

#Merge
data = pd.merge(location, price, how = "inner", on = ["id", "id"]) 


# In[4]:


data.head()


# In[5]:


#descibe
print(data.shape)
print(data.shape[0])
print(data.shape[1])
print(len(data.index))
print(len(data.columns))

data.describe()


# In[6]:


#Create dummies: 
dummy = pd.get_dummies(data["SType"]) #method to create new dummy dataframe
data = pd.concat([data, dummy], axis = 1) #Combine first and dummy dataframe


# In[7]:


data["Freehold"] = 0 #Create new column
data["Freehold"] = np.where(data.Tenure == "Freehold", 1, 0) #Condition to create dummy variable


# In[8]:


print(data.head())


# In[9]:


data.columns = data.columns.str.replace(" ", "") #Removes all of the spacing in column names


# In[10]:


####Question 1

#Linear regresion of transaction price on floor area

model_1 = ols("myValue ~ AreaSqft", data = data).fit(cov_type="HC1") #cov_type="HC1" implies heteroscedasticity robust standard error

#Obtain the exact coefficients
print("Model_1 coefficients:",model_1.params, sep='\n')
print('\n')
print("Model_1 standard errors:",model_1.bse, sep='\n')

results_summary = model_1.summary()

logging.info(results_summary)


print(results_summary)


# In[11]:


#Linear regresion of transaction price on floor area, “newsale”, “resale”.
model_2 = ols("myValue ~ AreaSqft + NewSale + Resale", data = data).fit(cov_type="HC1") 
print("Model_2 coefficients:",model_2.params, sep='\n')
print("Model_2 standard errors:",model_2.bse, sep='\n')


logging.info(model_2.summary())

results_summary = model_2.summary()

print(results_summary)


# In[12]:


#Linear regresion of transaction price on floor area, “newsale”, “resale”, Freehold
model_3 = ols("myValue ~ AreaSqft + NewSale + Resale + Freehold", data = data).fit(cov_type="HC1") 
print("Model_3 coefficients:",model_3.params, sep='\n')
print("Model_3 standard errors:",model_3.bse, sep='\n')


logging.info(model_3.summary())

results_summary = model_3.summary()


# In[13]:


#Linear regresion of transaction price on floor area, “newsale”, “resale”, Freehold along with dummies of salesmonth and District
model_4 = ols("myValue ~ AreaSqft + NewSale + Resale + Freehold + C(salesmonth) + C(District) ", data = data).fit(cov_type="HC1") 
print("Model_4 coefficients:",model_4.params, sep='\n')
print("Model_4 standard errors:",model_4.bse, sep='\n')


logging.info(model_4.summary())

results_summary = model_4.summary()

print(results_summary)


# In[14]:


#######Question 2
#Scatter plot between transaction price (Y) and floor area (X) and OLS fitted line
X = data["AreaSqft"]
Y = data["myValue"]
Y_pred = model_1.predict(X)

fig1 = plt.figure(figsize=(10,5))
plt.scatter(X, Y, color = "navy", label = "myValue")
plt.plot(X, Y_pred, color = "maroon", label = "Fitted values")
plt.savefig('OLS.png')


# In[15]:


####Question 3
### Create a column post to label pre and post intervention period
data['post'] = 0
data['post'] = np.where(data.ContractDate >= "2018-07-01", 1, 0)

#creating the interaction term
data["Interaction"] = data["post"] * data["treat"]


# In[16]:


#Difference in Difference (without district fixed effects or month fixed effects)
diff_in_diff = ols("myValue ~ post + treat + post*treat ", data = data).fit(cov_type="HC1") 
print("Diff in Diff coefficients:",diff_in_diff.params, sep='\n')
print("Diff in Diff standard errors:",diff_in_diff.bse, sep='\n')


logging.info(diff_in_diff.summary())

results_summary = diff_in_diff.summary()

print(results_summary)


# In[17]:


###  DID regression with district fixed effects but without month fixed effects

diff_in_diff = ols("myValue ~ post + post*treat + C(District)", data = data).fit(cov_type="HC1") 
print("Diff in Diff coefficients:",diff_in_diff.params, sep='\n')
print("Diff in Diff standard errors:",diff_in_diff.bse, sep='\n')


logging.info(diff_in_diff.summary())

results_summary = diff_in_diff.summary()

print(results_summary)


# In[18]:


####DID regression with district fixed effects and month fixed effects

diff_in_diff = ols("myValue ~ treat + post*treat + C(District) + C(salesmonth)", data = data).fit(cov_type="HC1") 
print("Diff in Diff coefficients:",diff_in_diff.params, sep='\n')
print("Diff in Diff standard errors:",diff_in_diff.bse, sep='\n')


logging.info(diff_in_diff.summary())

results_summary = diff_in_diff.summary()

print(results_summary)


# In[19]:


#Question 4 
#Grouping by month and treatment then finding the mean of each group
#as_index = False means that I don't want the "salesmonth" and "treat" to become the index since I want to split them up
#https://stackoverflow.com/questions/41236370/what-is-as-index-in-groupby-in-pandas

newdata = data.groupby(["salesmonth", "treat"], as_index=False).agg({"myValue": "mean"})

#Splitting into two treatment and control
value1 = newdata.loc[newdata["treat"] == 1]
value2 = newdata.loc[newdata["treat"] == 0]

### trend of housing price in each month in 2018
fig1 = plt.figure(figsize=(10,5))
plt.plot(value1['salesmonth'],value1['myValue'],label='treatment')
plt.plot(value2['salesmonth'],value2['myValue'],label='control')
plt.legend()
plt.xlabel('Months (2018)')
plt.ylabel('Avg Transaction price in SGD')
#plt.show()
plt.savefig('Trend of Housing Price.png')


logging.critical('completed')


# In[ ]:





# In[ ]:





# In[ ]:




