#!/usr/bin/env python
# coding: utf-8

# 

# #  Predicting Whether The Account Holders Will Default Next Month. This Means We Are To Predict The Probability Of Individuals’ Willingness To Pay Back a Credit-Card Loan.

# ## Problem Statement
# 
# Our client is a credit card company. They have brought us a dataset that includes some demographics and recent financial data, over the past 6 months, for a sample of 30,000 of their account holders. This data is at the credit account level; in other words, there is one row for each account (you should always clarify what the definition of a row is, in a dataset). Rows are labelled by whether, in the next month after the 6-month historical data period, an account owner has defaulted, or in other words, failed to make the minimum payment.

# ##  Aim Of The Research
# 
# The research aims at predicting the credit card default beforehand, that is predicting whether an account will default next month and to identify the potential customer base that can be offered various credit instruments so as to invite minimum default.

# In[4]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import scipy
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


# reading dataset
df = pd.read_csv("default of credit card clients.csv", skiprows = 1) # skiprows to ommit first row.


# In[6]:


# viewing the data frame
df.head()


# In[7]:


# a view of what the column is about
df.columns


# In[8]:


#saving the columns as a list

account_holders = df.columns
account_holders


# In[9]:


# checking the data types
df.dtypes


# In[70]:


df.index


# In[6]:


# knowing the lenght of dataframe i.e the numbers of rows
len(df)


# In[72]:


# knowing the numbers of rows and columns
df.shape


# In[10]:


# changing the column case to uppercase

df.columns=df.columns.str.upper()


# In[74]:


df.columns


# In[11]:


# renaming column

df.columns=['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',  'PAY_0', 'PAY_2',  'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',  'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'DEFAULT_PAYMENT_NEXT_MONTH']


# In[12]:


#dropping the ID column

df.drop(['ID'], axis=1)


# In[9]:


df.head()


# In[20]:


df.describe()


# In[77]:


# an overview of the dataset

df.info()


# In[85]:


# checking through for duplicates
df.loc[df.duplicated()]


# In[9]:


# summing all the different column values
df.sum()


# In[24]:


# checking for null values

df.isnull().sum()


# # Data Visualization 

# #### This research employed a binary variable, default  payment  next  month (Yes = 1, No = 0), as the response variable.

# In[11]:


df['DEFAULT_PAYMENT_NEXT_MONTH'].value_counts()


# In[66]:


df['DEFAULT_PAYMENT_NEXT_MONTH'].value_counts().plot.bar()


# ### This means that out of 30000 Account Holders, 23364 people (around 78%) did not default payment.

# In[65]:


plt.figure(figsize=(10,10))
plt.pie(x=[6636, 23364], labels=['1','0'], autopct='%1.0f%%', pctdistance=0.5,labeldistance=0.7,colors=['y','b'])
plt.title('Distribution of DEFAULT_PAYMENT_NEXT_MONTH')


# As shown above, the default probability of the sample is 22%. What this means is that, 78% of account holders will not default next month, while 22% account holders will default next month. 

# ## Visualizing Categorical Features

# - SEX
# - EDUCATION
# - MARRIAGE
# - AGE

# In[43]:


df['SEX'].value_counts()


# In[44]:


df['EDUCATION'].value_counts()


# In[45]:


df['MARRIAGE'].value_counts()


# In[40]:


plt.figure(figsize=(18, 6))

plt.subplot(1,3,1)
plt1 = df.SEX.value_counts().plot(kind='bar')
plt.title('Sex Histogram')
plt1.set(xlabel = 'Sex', ylabel='Frequency of Sex')

plt.subplot(1,3,2)
plt1 = df.EDUCATION.value_counts().plot(kind='bar')
plt.title('Education Histogram')
plt1.set(xlabel = 'Education', ylabel='Frequency of Education')

plt.subplot(1,3,3)
plt1 = df.MARRIAGE.value_counts().plot(kind='bar')
plt.title('Marriage Histogram')
plt1.set(xlabel = 'Marriage', ylabel='Frequency of Marriage')


plt.show()


# ## Inference :
# 
# From the SEX Features (1 = male; 2 = female). It can be inferred that: 60% Account Holders in the dataset are Female Account while  40% Account Holders are Male.
# 
# From the Education Features (1 = graduate school; 2 = university; 3 = high school; 4 = others). It can be inferred that: The Account Holders in the dataset are more University, follwed by  Graduate School, High School and others. 
# 
# From the Marriage Features (1 = married; 2 = single; 3 = others).The Account Holders in the dataset that are 
#  are the Single, follwed by Married, and others. 
# 

# In[51]:


plt.figure(figsize=(18,5))

plt.subplot(1,3,1)
sns.countplot(x='SEX' ,hue='DEFAULT_PAYMENT_NEXT_MONTH', data=df,palette='plasma')

plt.subplot(1,3,2)
sns.countplot(x='EDUCATION',hue='DEFAULT_PAYMENT_NEXT_MONTH',data=df,palette='viridis')
plt.ylabel(' ')
plt.yticks([ ])

plt.subplot(1,3,3)
sns.countplot(x='MARRIAGE',hue='DEFAULT_PAYMENT_NEXT_MONTH',data=df,palette='copper')
plt.ylabel(' ')
plt.yticks([ ])


# ## Inference :
# 
# This research employed a binary variable, default payment (Yes = 1, No = 0) 
# 
# Making Comparism between Sex which represents Gender shows that a Female Individual has higher probablity of Defualting next month (probability of failing to make the minimum payment) than the Male individuals. 
# 
# Checkecking the results of Educational Categories where 1 = graduate school; 2 = university; 3 = high school; 4 = others. shows that, the University Category have the probality of defaulting next month (probability of failing to make the minimum payment)
# 
# Checkecking the results of Marriage Categories where 1 = married; 2 = single; 3 = others. Shows that, Married and Single individauls have the probality of defaulting next month (probability of failing to make the minimum payment) 

# In[24]:


df['AGE'].value_counts()


# In[47]:


plt.figure(figsize=(16,6))

plt.subplot(2,2,1)
plt.title('AGE')
sns.distplot(df.AGE)

plt.subplot(2,2,2)

plt.title('Age Spread')
sns.boxplot(y=df.AGE)
#plt.hist(df.DEFAULT_PAYMENT_NEXT_MONTH)

plt.show()


# In[66]:


corr = df.corr()
plt.figure(figsize=(18,8))
sns.heatmap(corr, annot = True, cmap='BuPu')


# ## Model Building 
# 

# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve


# In[58]:


target_name = 'DEFAULT_PAYMENT_NEXT_MONTH'
X = df.drop('DEFAULT_PAYMENT_NEXT_MONTH', axis=1) # dropping the target variable
robust_scaler = RobustScaler() # used to rescale features into same scale. eg, limit bal and age
X = robust_scaler.fit_transform(X) # where X is a variable that contains all the features 
y = df[target_name] # the target variable 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)


# In[59]:


# funtion for printing a good Confusion Matrics
def CMatrix(CM, labels=['NON_DEFAULT_PAYMENT_NEXT_MONTH', 'DEFAULT_PAYMENT_NEXT_MONTH']):
    df = pd.DataFrame(data=CM, index=labels, columns=labels)
    df.index.name='TRUE'
    df.columns.name='PREDICTION'
    df.loc['Total'] = df.sum()
    df['Total'] = df.sum()
    return df 


# ## DataFrame Preparation For Model Analysis

# In[60]:


# Data frame for metrics evalution
metrics = pd.DataFrame(index=['accuracy', 'precision', 'recall'],
                      columns=['NULL', 'ClassTree', 'NaiveBayes'])


# In[61]:


# the NULL Metrics predicts the most common category
y_pred_test = np.repeat(y_train.value_counts().idxmax(), y_test.size)
metrics.loc['accuracy', 'NULL'] = accuracy_score(y_pred=y_pred_test, y_true=y_test)
metrics.loc['precision', 'NULL'] = precision_score(y_pred=y_pred_test, y_true=y_test)
metrics.loc['recall', 'NULL'] = recall_score(y_pred=y_pred_test, y_true=y_test)

CM = confusion_matrix(y_pred=y_pred_test, y_true=y_test)
CMatrix(CM)


# In[63]:


from sklearn.tree import DecisionTreeClassifier # Import the estimator mdel

# creating an instance for the estimator
class_tree = DecisionTreeClassifier(min_samples_split=30, min_samples_leaf=10, random_state=10) 

# training the estimator with the training data
class_tree.fit(X_train, y_train)

# evaluating the model
y_pred_test = class_tree.predict(X_test)
metrics.loc['accuracy', 'Class_Tree'] = accuracy_score(y_pred=y_pred_test, y_true=y_test)
metrics.loc['precision', 'Class_Tree'] = precision_score(y_pred=y_pred_test, y_true=y_test)
metrics.loc['recall', 'Class_Tree'] = recall_score(y_pred=y_pred_test, y_true=y_test)

CM = confusion_matrix(y_pred=y_pred_test, y_true=y_test)
CMatrix(CM)


# In[56]:


from sklearn.naive_bayes import GaussianNB

NBC = GaussianNB()
NBC.fit(X_train, y_train)
y_pred_test = NBC.predict(X_test)
metrics.loc['accuracy', 'NaiveBayes'] = accuracy_score(y_pred=y_pred_test, y_true=y_test)
metrics.loc['precision', 'NaiveBayes'] = precision_score(y_pred=y_pred_test, y_true=y_test)
metrics.loc['recall', 'NaiveBayes'] = recall_score(y_pred=y_pred_test, y_true=y_test)

CM = confusion_matrix(y_pred=y_pred_test, y_true=y_test)
CMatrix(CM)


# In[57]:


#data frame for the metrics
1*metrics


# ### Using KNN ML Algorithm

# In[19]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# In[20]:


knn = KNeighborsClassifier(n_neighbors = 23)


# In[21]:


# preparing the input (x) and target output (y)
x,y = df.drop(['ID', 'DEFAULT_PAYMENT_NEXT_MONTH'], axis=1), df['DEFAULT_PAYMENT_NEXT_MONTH']


# In[22]:


# spliting the dataset that will be used for trainning and testing
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 1)


# In[23]:


# fiting using the train dataset
knn.fit(x_train,y_train)


# In[24]:


# making predictions 
prediction = knn.predict(x_test)


# In[14]:


# checking the accuracy 
print('With KNN (k=24) accuracy is: ', knn.score(x_test,y_test))

