#!/usr/bin/env python
# coding: utf-8

# In[106]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
df= pd.read_csv(r'C:\ML\map1.csv')
df1=pd.read_csv(r'C:\ML\map4.csv')
df2=pd.read_csv(r'C:\ML\map2.csv')
df.head()


# # Python Task 1

# In[64]:


#Question 1: Car Matrix Generation#

car_matrix = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)
car_matrix.values[[range(len(car_matrix))]*2] = 0
car_matrix.head()


# In[38]:


#Question 2: Car Type Count Calculation#

df['car_type'] = pd.cut(df['car'], bins=[float('-inf'), 15, 25, float('inf')],labels=['low', 'medium', 'high'], right=False)
type_counts = df['car_type'].value_counts().to_dict()
df.head()


# In[47]:


#Question 3: Bus Count Index Retrieval#

bus_mean = df['bus'].mean()
bus_indexes = df[df['bus'] > 2 * bus_mean].index.tolist()
bus_indexes.sort()


# In[56]:


##Question 4: Route Filtering##

average=df.groupby('route')['truck'].mean()
filtered_routes = average[average > 7].index.tolist()
filtered_routes.sort()
filtered_routes


# In[61]:


#Question 5: Matrix Value Modification#

df_modified = car_matrix.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)
df_modified_rounded = df_modified.round(1)
df_modified_rounded 


# In[76]:


#Question 6: Time Check#

df1.head()


# In[93]:


df1['start_timestamp'] = pd.to_datetime(df1['startDay'] + ' ' + df1['startTime'],errors='coerce')

df1['end_timestamp'] = pd.to_datetime(df1['endDay'] + ' ' + df1['endTime'],errors='coerce')
incorrect_timestamps = ~((df1['start_timestamp'].notna()) &
                         (df1['end_timestamp'].notna()) & 
                             (df1['start_timestamp'].dt.time == pd.Timestamp("00:00:00").time()) &
                             (df1['end_timestamp'].dt.time == pd.Timestamp("23:59:59").time()) &
                             (df1['end_timestamp'] > df1['start_timestamp']) &
                             (df1['start_timestamp'].dt.dayofweek.isin(range(7))) &
                             (df1['end_timestamp'].dt.dayofweek.isin(range(7))))


# # Python Task 2

# In[114]:


df2.head()


# In[115]:


#Question 1: Distance Matrix Calculation#

distance_matrix = df2.pivot(index='id_start', columns='id_end', values='distance').fillna(0)
distance_matrix = distance_matrix.add(distance_matrix.T, fill_value=0)
distance_matrix.values[[range(len(distance_matrix))]*2] = 0
cumulative_distance_matrix = distance_matrix.cumsum(axis=1)
cumulative_distance_matrix.fillna(0)


# In[120]:


#Question 2: Unroll Distance Matrix#

id_start_values = df2.index.tolist()
id_end_values = df2.columns.tolist()
id_start_unrolled = []
id_end_unrolled = []
distance_unrolled = []
for start_id in id_start_values:
        for end_id in id_end_values:
            if start_id != end_id:
                id_start_unrolled.append(start_id)
                id_end_unrolled.append(end_id)
                distance_unrolled.append(df2.loc[start_id, end_id])
unrolled_df = pd.DataFrame({
        'id_start': id_start_unrolled,
        'id_end': id_end_unrolled,
        'distance': distance_unrolled
    })
unrolled_df


# In[133]:


#Question 3: Finding IDs within Percentage Threshold#

unrolled_df['id_start'].mean()

def modify_value(val):
    if val > 20:
        return round(val * 0.75, 1)
    else:
        return round(val * 1.25, 1)

data = df2.applymap(modify_value)
data


# In[147]:


#Question 4: Calculate Toll Rate#


rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
for vehicle_type, rate_coefficient in rate_coefficients.items():
        df[vehicle_type] = df2['distance'] * rate_coefficient
df.head()


# In[159]:


#Question 5: Calculate Time-Based Toll Rates#

weekday_time_ranges = [(pd.Timestamp("00:00:00").time(), pd.Timestamp("10:00:00").time()),
                           (pd.Timestamp("10:00:00").time(), pd.Timestamp("18:00:00").time()),
                           (pd.Timestamp("18:00:00").time(), pd.Timestamp("23:59:59").time())]

weekend_time_ranges = [(pd.Timestamp("00:00:00").time(), pd.Timestamp("23:59:59").time())]
weekday_factors = [0.8, 1.2, 0.8]
weekend_factor = 0.7


# In[ ]:





# In[ ]:





# In[ ]:




