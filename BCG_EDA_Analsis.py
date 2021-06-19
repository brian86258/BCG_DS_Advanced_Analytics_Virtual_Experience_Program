#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# # BCG EDA 

# ![image.png](attachment:image.png)

# In[2]:


train_data = pd.read_csv('ml_case_training_data.csv')
train_hist_data = pd.read_csv('ml_case_training_hist_data.csv')
output_data = pd.read_csv('ml_case_training_output.csv')


# In[3]:


# Merge the 'churn' to train data
full_train_data = train_data.merge(output_data, on='id')


# In[4]:


train_data.shape


# In[5]:


full_train_data.shape


# In[6]:


total_null = full_train_data.isnull().sum().sort_values(ascending = False)
null_percentage = (full_train_data.isnull().sum() / full_train_data.isnull().count()).sort_values(ascending = False)

missing_data = pd.concat([total_null, null_percentage],keys=['Total_Null','Null_percentage'], axis = 1)


# In[7]:


missing_data.head(20)


# In[ ]:





# ## We can see that there are multiple variable present too much null valuse, which cannot provide vital information. 
# General should not exceed 15% null percentage

# In[8]:


# these are the variables contains too much null values, which may 
drop_categories = missing_data[missing_data['Null_percentage'] > 0.15].index
drop_categories


# In[9]:


# data: data after drop too much null values
data = full_train_data.drop(drop_categories, axis = 1)


# In[10]:


missing_data.loc[data.columns].sort_values(by='Null_percentage',ascending = False)


# ## deal with null values

# In[11]:


# find the numeric columns
num_col = data._get_numeric_data().columns.tolist()
cat_col = set(data.columns) - set(num_col)
print("Num cols :{}, \n\nCat cols: {}".format(num_col, cat_col))


# In[12]:


# Replace null with medain for numeric values
for col in num_col:
    data[col] = data[col].fillna(data[col].median())


# In[13]:


data[num_col].isnull().sum()


# In[14]:


# fill categoriacal column by adding an new "Unknown" Category
# Doing so is due to there are some variable such as date_end, if fill with mode may result in date_end prior to activ, which not make sense 
for col in cat_col:
    print(col)
    data[col].fillna('Unknown', inplace=True)


# In[15]:


data[cat_col].isnull().sum()


# In[16]:


data.isnull().sum()


# ## EDA

# ### Detect Outliers

# In[17]:


def detect_outlier(df, col):
    print(col)
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    l_outlier = df[col].apply(lambda x: x <= lower_bound).sum()
    u_outlier = df[col].apply(lambda x: x >= upper_bound).sum()
    print("lower outlier :{}, Upper outliers: {}".format(l_outlier,u_outlier))

    


# In[18]:


for col in num_col:
    detect_outlier(data, col)


# In[19]:


data.shape


# ### Churned vs Non-Churned customers:
# Consumption wise:
# * cons_12m: electricity consumption of the past 12 months
# * cons_gas_12m: gas consumption of the past 12 months
# * cons_last_month: electricity consumption of the last month
# 
# ---
# Relationship wise:
# * net_margin: total net margin
# * num_years_antig: antiquity of the client (in number of years)
# 

# In[20]:


# Churn Percentage
churn_percentage = data['churn'].sum() / data['churn'].count()
print("Churn Percentage: {:.2}%".format(churn_percentage))


# In[21]:


data[['cons_12m','churn']].groupby(['churn'], as_index = False).mean().sort_values(by='cons_12m', ascending=False)


# In[22]:


data[['forecast_cons_12m','churn']].groupby(['churn'], as_index = False).mean().sort_values(by='forecast_cons_12m', ascending=False)


# In[23]:


data[['forecast_cons_year','churn']].groupby(['churn'], as_index = False).mean().sort_values(by='forecast_cons_year', ascending=False)


# From Consumption wise, we can find most of customer churn due to the uprise in forecast consumption  
# 
# As for non-churned customers, mostly already devote lots of resources in the past.
# 
# ----

# In[24]:


data[['net_margin','churn']].groupby(['churn'], as_index = False).mean().sort_values(by='net_margin', ascending=False)


# In[25]:


data[['num_years_antig','churn']].groupby(['churn'], as_index = False).mean().sort_values(by='num_years_antig', ascending=False)


# ## model

# In[27]:


import seaborn as sns

corrmat = data.corr()
top_corr_features = corrmat[abs(corrmat['churn'] > 0.02)].index
plt.figure(figsize=(9,9))
# Use these attributes to form the heatmap
# train_data[top_corr_features]
g = sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="coolwarm")


# In[29]:


data.shape


# In[30]:


# drop duplicate data entries
data = data.T.drop_duplicates().T


# In[31]:


data.shape


# ## Model

# ### Utilize Lanel Encoder to Encode Data

# In[32]:


from sklearn.preprocessing import LabelEncoder
labelcoder = LabelEncoder()


# In[33]:


for col in cat_col:
    data[col] = labelcoder.fit_transform(data[col])


# In[37]:


churn = data['churn']
churn=churn.astype('int')


# In[40]:


train_data = data.drop(['churn'], axis = 1)


# In[41]:


train_data.head()


# In[60]:


from sklearn.model_selection import train_test_split
# X_train,X_test,Y_train,Y_test = train_test_split(one_hot_ticket,Survived, test_size=.30)
X_train,X_test,Y_train,Y_test = train_test_split(train_data,churn, test_size=.30)


# In[46]:


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[62]:


# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_test, Y_test) * 100, 2)

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_test, Y_test) * 100, 2)
acc_sgd

# Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_test, Y_test) * 100, 2)
acc_svc

# KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_test, Y_test) * 100, 2)
acc_knn

# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_test, Y_test) * 100, 2)
acc_gaussian

# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_test, Y_test) * 100, 2)
acc_perceptron
# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_test, Y_test) * 100, 2)
acc_linear_svc

# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_test, Y_test) * 100, 2)
acc_decision_tree

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_test, Y_test) * 100, 2)
acc_random_forest


# ## Model Comparison:
# We can see using Random Forest has the highest score

# In[99]:


# drop string
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Accuracy_Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Accuracy_Score', ascending=False)


# ## Dimension Reduction

# In[94]:


from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[95]:


myPCA = PCA(10)
mySVD = TruncatedSVD(10)
myLDA = LinearDiscriminantAnalysis(10)


# In[83]:


myPCA.fit(X_train)


# In[84]:


RX_train = myPCA.transform(X_train)
RX_test = myPCA.transform(X_test)


# ## PCA

# In[86]:


# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_test, Y_test) * 100, 2)
acc_random_forest


# In[87]:


acc_random_forest


# ## SVD

# In[88]:


mySVD.fit(X_train)


# In[89]:


RX_train = mySVD.transform(X_train)
RX_test = mySVD.transform(X_test)


# In[101]:


# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_test, Y_test) * 100, 2)
acc_random_forest


# ## We can observe that using SVD and PCA did'nt help at the accuracy of the data

# In[100]:


from sklearn import metrics


# In[102]:


print('Precision:', metrics.precision_score(Y_test, Y_pred))
print('Recall:', metrics.recall_score(Y_test, Y_pred))
print('F1:', metrics.f1_score(Y_test, Y_pred))


# In[104]:


fpr, tpr, threshold = metrics.roc_curve(Y_test, Y_pred)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:




