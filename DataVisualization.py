#!/usr/bin/env python
# coding: utf-8

# # Forest Fire Size Prediction
# ## Data Visualization

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

# In[6]:


# Extract day, month, year from discovery clean date
df['disc_clean_date'] = pd.to_datetime(df['disc_clean_date'])

df['disc_month'] = df['disc_clean_date'].dt.month

# Drop the columns which are not required
df = df.drop(['disc_clean_date', 'disc_date_pre',               'wstation_byear', 'wstation_eyear'],axis=1)


# ### Data Visualization

# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster, HeatMap
from folium import Choropleth


# In[8]:


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

# In[9]:


plt.figure(figsize=(10, 5))
plt.title("Boxplot for Wind Variables", fontsize=20)
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Create boxplot
df.boxplot(column=["Wind_pre_30", "Wind_pre_15", "Wind_pre_7"], grid=False)

plt.show()


# In[10]:


plt.figure(figsize=(10, 5))
plt.title("Boxplot for Temperature Variables", fontsize=20)
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Create boxplot
df.boxplot(column=["Temp_pre_30", "Temp_pre_15", "Temp_pre_7"], grid=False)

plt.show()


# In[11]:


plt.figure(figsize=(10, 5))
plt.title("Boxplot for Humidity Variables", fontsize=20)
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Create boxplot
df.boxplot(column=["Hum_pre_30", "Hum_pre_15", "Hum_pre_7"], grid=False)

plt.show()


# In[12]:


plt.figure(figsize=(10, 5))
plt.title("Boxplot for Precipitation Variables", fontsize=20)
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Create boxplot
df.boxplot(column=["Prec_pre_30", "Prec_pre_15", "Prec_pre_7"], grid=False)

plt.show()


# In[13]:


# Group by 'disc_month' and calculate the average fire size for each month
average_fire_size_per_month = df.groupby('disc_month')['fire_size'].mean().reset_index()

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(average_fire_size_per_month['disc_month'], average_fire_size_per_month['fire_size'], color='red')
plt.xlabel('Month')
plt.ylabel('Average Fire Size')
plt.title('Average Fire Size by Month')
plt.show()

