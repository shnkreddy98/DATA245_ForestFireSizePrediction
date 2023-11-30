#!/usr/bin/env python
# coding: utf-8

# # Forest Fire Size Prediction

# ### Importing all the required libraries.

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import category_encoders as ce


# ### Reading the CSV files

# In[3]:


df = pd.read_csv("data/FW_Veg_Rem_Combined.csv")
df.head(5)


# ### Removing the redundant and unnecessary columns

# In[4]:


df = df.drop(['Unnamed: 0.1', 'Unnamed: 0', 'fire_name', 'state', 'cont_clean_date',
         'discovery_month', 'disc_date_final', 'cont_date_final', 'putout_time', 'disc_pre_year', 'disc_pre_month',
         'wstation_usaf', 'dstation_m', 'wstation_wban', 'fire_mag', 'weather_file'],axis=1)


# #### Removed columns with null values, redundant columns like fire_mag, fire_size and date variables

# In[5]:


df.head(5)


# #### As the target class is unbalanced we decided to club the smaller group of classes as 1.
# #### Clubbing (C,D,E,F,G) as (1) class and A,B as (0) class.
# #### 0 idicates small fire <25 Acres and 1 represents a widespread fire >25Acres.

# In[6]:


class_mapping = {'A': 0, 'B': 0, 'C':1, 'D':1, 'E':1, 'F':1, 'G':1}
df = df.replace(class_mapping)


# In[7]:


df.fire_size_class.value_counts()


# In[8]:


(df.fire_size_class.value_counts()/df.shape[0])*100


# ### Extracting the date and month and removing the redundant columns.

# In[9]:


# Extract day, month, year from discovery clean date
df['disc_clean_date'] = pd.to_datetime(df['disc_clean_date'])

df['disc_month'] = df['disc_clean_date'].dt.month

# Drop the columns which are not required
df = df.drop(['disc_clean_date', 'disc_date_pre',               'wstation_byear', 'wstation_eyear'],axis=1)


# In[10]:


df= df.drop(["fire_size"], axis = 1)


# In[11]:


df['Vegetation'] = df['Vegetation'].astype(object)


# ### Applying MinMaxScaler to the weather variables

# In[12]:


from sklearn.preprocessing import MinMaxScaler

trans = MinMaxScaler()
df.iloc[:, 5:21] = trans.fit_transform(df.iloc[:, 5:21])


# In[13]:


X = df.drop('fire_size_class',axis=1)
y = df['fire_size_class']


# In[14]:


X.columns


# ### Defining the function for target encoding

# In[15]:


def target_encode_multiclass(X,y): #X,y are pandas df and series
    y=y.astype(str)   #convert to string to onehot encode
    enc=ce.OneHotEncoder().fit(y)
    y_onehot=enc.transform(y)
    class_names=y_onehot.columns  #names of onehot encoded columns
    X_obj=X.select_dtypes('object') #separate categorical columns
    X=X.select_dtypes(exclude='object')
    for class_ in class_names:
        enc=ce.TargetEncoder()
        enc.fit(X_obj,y_onehot[class_]) #convert all categorical
        temp=enc.transform(X_obj)       #columns for class_
        temp.columns=[str(x)+'_'+str(class_) for x in temp.columns]
        X=pd.concat([X,temp],axis=1)    #add to original dataset

    return X


# In[16]:


X = target_encode_multiclass(X,y)


# In[17]:


from sklearn.metrics import roc_curve, accuracy_score
from sklearn.metrics import auc, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


# ### Random Forest

# In[18]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_folds = 5

# Set up k-fold cross-validation
stratified_kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)


# In[19]:


RF = RandomForestClassifier()

# Lists to store metrics for each fold
RF_precision_list = []
RF_recall_list = []
RF_f1_list = []

for train_index, val_index in stratified_kf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        
    RF.fit(X_train_fold, y_train_fold)
    RF_pred = RF.predict(X_val_fold)

    # Calculate metrics for each fold
    classification_report_fold = classification_report(y_val_fold, RF_pred, output_dict=True)
    
    RF_precision_list.append(classification_report_fold['weighted avg']['precision'])
    RF_recall_list.append(classification_report_fold['weighted avg']['recall'])
    RF_f1_list.append(classification_report_fold['weighted avg']['f1-score'])

# Calculate mean metrics across all folds
mean_RF_precision = np.mean(RF_precision_list)
mean_RF_recall = np.mean(RF_recall_list)
mean_RF_f1 = np.mean(RF_f1_list)

# Print or use the mean metrics as needed
print(f'Mean Precision for RF: {mean_RF_precision}')
print(f'Mean Recall for RF: {mean_RF_recall}')
print(f'Mean F1-Score for RF: {mean_RF_f1}')


# In[20]:


import pickle

Pkl_Filename = "RandomForest_ForestFire.pkl"

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(RF, file)

