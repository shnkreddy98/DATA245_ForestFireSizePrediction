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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import category_encoders as ce


# ### Reading the CSV files

# In[3]:


df = pd.read_csv("data/FW_Veg_Rem_Combined.csv")
df.head(5)


# In[4]:


df.describe(include = "all")


# In[5]:


df.info()


# #### Data Dictionary

# - Fire_size_class - Class of Fire Size (A-G)
# - Stat_cause_descr - Cause of Fire
# - Latitude - Latitude of Fire
# - Longitude - Longitude of Fire
# - Discovery_month - Month in which Fire was discovered
# - Vegetation - Dominant vegetation in the areas (can save some factors of vegetation) 
# - Temp_pre - temperature in deg C at the location of fire up to 30, 15 and 7 days prior
# - Temp_cont - temperature in deg C at the location of fire up to day the fire was 
# - Wind_pre - wind in deg C at the location of fire up to 30, 15 and 7 days prior
# - Wind_cont - wind in deg C at the location of fire up to day the fire was 
# - Prec_pre - Precipitation in deg C at the location of fire up to 30, 15 and 7 days prior
# - Prec_cont - Precipitation in deg C at the location of fire up to day the fire was 
# - Hum_pre - Humidity in deg C at the location of fire up to 30, 15 and 7 days prior
# - Hum_cont - Humidity in deg C at the location of fire up to day the fire was 
# - Remoteness - non-dimensional distance to closest city

# ### Removing the redundant and unnecessary columns

# In[6]:


df = df.drop(['Unnamed: 0.1', 'Unnamed: 0', 'fire_name', 'state', 'cont_clean_date',
         'discovery_month', 'disc_date_final', 'cont_date_final', 'putout_time', 'disc_pre_year', 'disc_pre_month',
         'wstation_usaf', 'dstation_m', 'wstation_wban', 'fire_mag', 'weather_file'],axis=1)


# #### Removed columns with null values, redundant columns like fire_mag, fire_size and date variables

# In[7]:


df.head(5)


# ### Data Visualization

# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster, HeatMap
from folium import Choropleth


# In[9]:


# Create a base map centered at a specific location
map_center = [df['latitude'].mean(), df['longitude'].mean()]
mymap = folium.Map(location=map_center, zoom_start=4)

# Create a MarkerCluster to group markers at the same location
marker_cluster = MarkerCluster().add_to(mymap)

# Add markers for each data point
for index, row in df.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"Fire Size Class: {row['fire_size_class']}",
        icon=None,  # You can customize the icon if needed
    ).add_to(marker_cluster)

# Display the map in the notebook
display(mymap)


# #### The dataset has fire occurances data from all over US, concentrated in california and Southeastern part.

# In[10]:


plt.figure(figsize=(10, 5))
plt.title("Boxplot for Wind Variables", fontsize=20)
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Create boxplot
df.boxplot(column=["Wind_pre_30", "Wind_pre_15", "Wind_pre_7"], grid=False)

plt.show()


# In[11]:


plt.figure(figsize=(10, 5))
plt.title("Boxplot for Temperature Variables", fontsize=20)
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Create boxplot
df.boxplot(column=["Temp_pre_30", "Temp_pre_15", "Temp_pre_7"], grid=False)

plt.show()


# In[12]:


plt.figure(figsize=(10, 5))
plt.title("Boxplot for Humidity Variables", fontsize=20)
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Create boxplot
df.boxplot(column=["Hum_pre_30", "Hum_pre_15", "Hum_pre_7"], grid=False)

plt.show()


# In[13]:


plt.figure(figsize=(10, 5))
plt.title("Boxplot for Precipitation Variables", fontsize=20)
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Create boxplot
df.boxplot(column=["Prec_pre_30", "Prec_pre_15", "Prec_pre_7"], grid=False)

plt.show()


# #### Will apply minmaxscaler to take care of these outliers in the weather data

# In[14]:


(df.fire_size_class.value_counts()/df.shape[0])*100


# #### As the target class is unbalanced we decided to club the smaller group of classes as 1.
# #### Clubbing (C,D,E,F,G) as (1) class and A,B as (0) class.
# #### 0 idicates small fire <25 Acres and 1 represents a widespread fire >25Acres.

# In[15]:


class_mapping = {'A': 0, 'B': 0, 'C':1, 'D':1, 'E':1, 'F':1, 'G':1}
df = df.replace(class_mapping)


# In[16]:


df.fire_size_class.value_counts()


# In[17]:


(df.fire_size_class.value_counts()/df.shape[0])*100


# ### Extracting the date and month and removing the redundant columns.

# In[18]:


# Extract day, month, year from discovery clean date
df['disc_clean_date'] = pd.to_datetime(df['disc_clean_date'])

df['disc_month'] = df['disc_clean_date'].dt.month

# Drop the columns which are not required
df = df.drop(['disc_clean_date', 'disc_date_pre',               'wstation_byear', 'wstation_eyear'],axis=1)


# In[19]:


# Group by 'disc_month' and calculate the average fire size for each month
average_fire_size_per_month = df.groupby('disc_month')['fire_size'].mean().reset_index()

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(average_fire_size_per_month['disc_month'], average_fire_size_per_month['fire_size'], color='red')
plt.xlabel('Month')
plt.ylabel('Average Fire Size')
plt.title('Average Fire Size by Month')
plt.show()


# ### Bigger fires occur during the summer season, so months columns will be an important variable

# In[20]:


df= df.drop(["fire_size"], axis = 1)


# In[21]:


df['Vegetation'] = df['Vegetation'].astype(object)


# ### Applying MinMaxScaler to the weather variables

# In[22]:


from sklearn.preprocessing import MinMaxScaler

trans = MinMaxScaler()
df.iloc[:, 5:21] = trans.fit_transform(df.iloc[:, 5:21])


# In[23]:


X = df.drop('fire_size_class',axis=1)
y = df['fire_size_class']


# In[24]:


X.columns


# ### Defining the function for target encoding

# In[25]:


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


# In[26]:


X = target_encode_multiclass(X,y)


# In[27]:


X.info()


# In[28]:


sns.heatmap(pd.concat([X, y], axis = 1).corr())
plt.show()


# In[29]:


X.describe(include="all")


# In[30]:


from sklearn.metrics import roc_curve, accuracy_score
from sklearn.metrics import auc, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


# ### Calculating Accuracy for 5 folds

# In[31]:


X = X.values
y = y.values

# Define the number of folds for cross-validation
num_folds = 5

# Initialize StratifiedKFold for stratified sampling based on the distribution of classes
stratified_kfold = StratifiedKFold(n_splits=num_folds, 
                                   shuffle=True, 
                                   random_state=42)


# In[32]:


classifiers=[['Logistic Regression :',LogisticRegression()],
             ['Decision Tree Classification :',DecisionTreeClassifier()],
             ['Random Forest Classification :',RandomForestClassifier()],
             ['K-Neighbors Classification :',KNeighborsClassifier()],
             ['Gausian Naive Bayes :',GaussianNB()],
             ['Support Vector Classification :',SVC()]]

cla_pred=[]

for name,model in classifiers:
    model=model
    # Perform K-fold cross-validation
    for fold, (train_index, test_index) in enumerate(stratified_kfold.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model.fit(X_train, y_train)

        # Make predictions on the test set
        predictions = model.predict(X_test)
        
        # Evaluate the model
        cla_pred.append(accuracy_score(y_test,predictions))
        print(name, fold+1, accuracy_score(y_test,predictions))


# ### Printing Classifiction Report for 5 folds

# #### Logistic Regression

# In[33]:


X1 = df.drop('fire_size_class',axis=1)
y1 = df['fire_size_class']

X1 = target_encode_multiclass(X1, y1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

# Set up k-fold cross-validation
stratified_kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)


# In[34]:


LR = LogisticRegression()

# Lists to store metrics for each fold
LR_precision_list = []
LR_recall_list = []
LR_f1_list = []

for train_index, val_index in stratified_kf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        
    LR.fit(X_train_fold, y_train_fold)
    LR_pred = LR.predict(X_val_fold)

    # Calculate metrics for each fold
    classification_report_fold = classification_report(y_val_fold, LR_pred, output_dict=True)
    
    LR_precision_list.append(classification_report_fold['weighted avg']['precision'])
    LR_recall_list.append(classification_report_fold['weighted avg']['recall'])
    LR_f1_list.append(classification_report_fold['weighted avg']['f1-score'])

# Calculate mean metrics across all folds
mean_LR_precision = np.mean(LR_precision_list)
mean_LR_recall = np.mean(LR_recall_list)
mean_LR_f1 = np.mean(LR_f1_list)

# Print or use the mean metrics as needed
print(f'Mean Precision for LR: {mean_LR_precision}')
print(f'Mean Recall for LR: {mean_LR_recall}')
print(f'Mean F1-Score for LR: {mean_LR_f1}')


# #### DecisionTree

# In[35]:


DT = DecisionTreeClassifier()

# Lists to store metrics for each fold
DT_precision_list = []
DT_recall_list = []
DT_f1_list = []

for train_index, val_index in stratified_kf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        
    DT.fit(X_train_fold, y_train_fold)
    DT_pred = DT.predict(X_val_fold)

    # Calculate metrics for each fold
    classification_report_fold = classification_report(y_val_fold, DT_pred, output_dict=True)
    
    DT_precision_list.append(classification_report_fold['weighted avg']['precision'])
    DT_recall_list.append(classification_report_fold['weighted avg']['recall'])
    DT_f1_list.append(classification_report_fold['weighted avg']['f1-score'])

# Calculate mean metrics across all folds
mean_DT_precision = np.mean(DT_precision_list)
mean_DT_recall = np.mean(DT_recall_list)
mean_DT_f1 = np.mean(DT_f1_list)

# Print or use the mean metrics as needed
print(f'Mean Precision for DT: {mean_DT_precision}')
print(f'Mean Recall for DT: {mean_DT_recall}')
print(f'Mean F1-Score for DT: {mean_DT_f1}')


# #### Random Forest

# In[36]:


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


# #### KNN

# In[37]:


RF = KNeighborsClassifier()

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


# #### Naive Bayes

# In[38]:


RF = GaussianNB()

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


# #### SVM

# In[39]:


RF = SVC()

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


# ### Plotting the ROC curves for all 6 models

# In[40]:


models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
#     'SVM': SVC(probability=True)
}

# Set up k-fold cross-validation
stratified_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Iterate through each model
for model_name, model in models.items():
    print(f"Evaluating {model_name}")
    
    # Perform k-fold cross-validation
    mean_fpr = np.linspace(0, 1, 100)
    tpr_list = []
    
    for train_index, val_index in stratified_kf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict_proba(X_val_fold)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_val_fold, y_pred)
        tpr_list.append(np.interp(mean_fpr, fpr, tpr))
    
    # Compute mean and standard deviation of the ROC curves
    mean_tpr = np.mean(tpr_list, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(tpr_list, axis=0)
    
    # Plot the training ROC curve with mean and standard deviation
    plt.plot(mean_fpr, mean_tpr, label=f'{model_name} (AUC = {mean_auc:.2f} ± {np.mean(std_auc):.2f})')


# Plot the random chance line
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Chance')

# Set plot labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Training ROC Curves')
plt.legend(loc='lower right')
plt.show()


# ### select models for ensemble based on roc curves, 3 or all?

# #### Ensemble Method
# 

# In[41]:


from sklearn.metrics import log_loss
from sklearn.ensemble import VotingClassifier, StackingClassifier

LR = LogisticRegression()
DT = DecisionTreeClassifier()
RF = RandomForestClassifier()
KNN = KNeighborsClassifier()
SVM = SVC()
NB = GaussianNB()

# # Voting Classifier
voting_classifier = VotingClassifier(estimators=[('rf', LR), 
                                                 ('knn', KNN), 
                                                 ('lr', LR)],
                                     voting='hard')
# 'hard' for majority voting, 
#'soft' for weighted voting based on probabilities


# Stacking Classifier
stacking_classifier = StackingClassifier(estimators=[('rf', LR), 
                                                     ('knn', KNN), 
                                                     ('lr', LR)],
                                         final_estimator=LogisticRegression())

# Train and evaluate each classifier
classifiers = [voting_classifier, 
               stacking_classifier]

for classifier in classifiers:
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{classifier.__class__.__name__} Accuracy: {accuracy}")


# ### Random forest works better, followed by KNN, Logistic Regression, Decision Tree and SVM in that order
