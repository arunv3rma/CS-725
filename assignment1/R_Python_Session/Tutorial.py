# coding: utf-8

# # Getting started with Python for data science challenges.

# ## Why python ?
# * Because of its rich libraries. [this does not mean others dont have ]
# * You can use Python when your data analysis tasks need to be integrated with web apps.
# * Growing user base.
# 
# There are few disadvantages of Python if you compare it with contenders like R. To qoute one,
# R's visulization libraries are better but Python's visualization libraries like Seaborn are filling up this gap.
# 
# So it's upto you on which one you would go for. 

# ## Getting started with the tutorial.
# * We will be using Kaggle's <a href="https://www.kaggle.com/c/titanic">titanic challenge</a> as a reference for the tutorial. 
# * We will go over
#     * How to read data.
#     * Basics of Numpy and Pandas.
#     * How to use basic models defined in Sci-Kit learn library.
#     
# 

# ## Reading Data using Pandas.
# 

# In[33]:

# import required library
import pandas as pd # now you can refer to pandas library as 'pd' from here on.
import numpy as np

df = pd.read_csv('train.csv', header=0)
df
# Below is the whole data frame. Pandas loads the data as into a frame which is basically similar to that of a table 
# in RDBMS.


# In[18]:

#to choose top 5 rows we use head (similarly tail for last 5.)
df.head(5)


# In[19]:

# to know the data types of all the columns in the data frame we use
df.dtypes


# In[20]:

# to know info about the number of rows and null values we use
df.info()


# In[21]:

# to know the basic statistics about the data frame. [report only for numerical columns.]
df.describe()
#you could know stats like min of a column and others.


# ## Referencing and indexing in Pandas

# In[22]:

#show first 10 age column values. syntax: df['column_name'][row_values] or df.column_name['row_vlaues']
df['Age'][0:10] #returns a series (series is a pandas variant of tupple.)


# In[23]:

#using above to compute mean of Age column
df['Age'].mean()
#this ignores 'Nan' Values.


# In[24]:

# to select multiple columns at a time we provide the list of column names instead of one value
df[['Sex','Pclass','Age']][0:10]
#df[['Sex','Pclass','Age']] would return all the 891 rows.


# In[25]:

# one of the most used operation is filtering.
# it is similar to that of SQL's where cluase.
# to see the passenger details whose age is greater than 70
df[df['Age']>70] #df[condition]


# In[26]:

#projecting certain columns of the selection. 
df[df['Age'] > 60][['Sex', 'Pclass', 'Age', 'Survived']]
# we can observe that males in class 1 are most likely to die. 


# In[27]:

# We need to filter null values before we could apply any algorithm.
# to choose null values we use the "where" filtering as below.
df[df['Age'].isnull()][['Sex', 'Pclass', 'Age']][0:10]
#you could use a negation in the condition to get only rows with non null values.


# In[28]:

# Applying multiple conditions at a time. (Below computes number of passengers in each class.)
for i in range(1,4):
    print(i, len(df[ (df['Sex'] == 'male') & (df['Pclass'] == i) ]))


# In[29]:

#visulizing data using Pandas's histogram
import pylab as P
#plot histogram on Age column but before that we drop 'Nan' values.
#basic command df['Column_name'].hist()
df['Age'].dropna().hist(bins=16, range=(0,80), alpha = .5)
P.show()


# ## Formating and Cleaning data 

# In[34]:

# Not all features are in the required format. Ex: Gender is in string format(female,male). 
# We shall make it into nominal values.
# we add a new column 'Gender' 
#df['Gender'] = df['Sex'].map( lambda x: x[0].upper() )
df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
# to check the modification we use the describe method. 
# we see that all the 891 rows have been updated correctly.
df.describe()
#print(df.head())


# In[31]:

df.head()


# In[35]:

#filling null values with some appropriate values

#we fill age's Nan values with median age for given gender and passenger class.

# first we use numpy's array to compute median ages for given classes.

median_ages = np.zeros((2,3))
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = df[(df['Gender'] == i) &  (df['Pclass'] == j+1)]['Age'].dropna().median()
        
#median_ages
#to avoid losing original data we make a copy of it
df['AgeFill'] = df['Age']
df[ df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)


# In[36]:

# we now fill each Nan value of 'AgeFill' column with respective classes's median.
for i in range(0, 2):
    for j in range(0, 3):
        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),                'AgeFill'] = median_ages[i,j]
df[ df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)


# ## Feature Engineering.

# In[37]:

# we can try impact of different combinations of features
df['Age*Class'] = df.AgeFill * df.Pclass
import pylab as p
df['Age*Class'].hist()
p.show()


# In[38]:

# Dropping columns
df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1) 
# Dropping rows with Nan values.
df = df.dropna()


# In[39]:

df.describe()


# In[40]:

# to look at the columns of the data frame.
df.columns


# ## Fitting a linear model.

# In[41]:

#We first split available training data into validation and training data.
train_data=df.ix[0:500,]
validate_data=df.ix[500:,]


# In[42]:

#importing sklearn library.
from sklearn import linear_model

#create a linear regression model
lr=linear_model.LinearRegression()
trainx=train_data.ix[:,2:]
trainy=train_data.ix[:,1]

#fit the linear model.
lr.fit(trainx,trainy)
#access the coeffcients using 
print("w values for the linear model",lr.coef_)

testx=validate_data.ix[:,2:]
testy=validate_data.ix[:,1]
ypredicted=lr.predict(testx)
predicted_values=pd.DataFrame(ypredicted)

def fun1(x):
    if x>0.5:
        return 1
    else:
        return 0

predicted_labels = predicted_values[0].apply(fun1)

print(predicted_labels.head())
print(testy.head())

