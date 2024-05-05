# To import the libraries and data:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn_extra.cluster import KMedoids
from scipy.cluster.hierarchy import linkage,dendrogram,fcluster
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("C:\\Users\\Ziad\\Downloads\\Downloads\\Just Study!\\Data Mining\\project\\Financial_ Application_ Behavior_ Dataset.csv")

# To find and rename the unnamed coulumns:
unnamed_cols = data.columns[data.columns.str.contains('Unnamed')]

if len(unnamed_cols) > 0:
    print("Unnamed columns found:", unnamed_cols.tolist())
else:
    print("No unnamed columns found.")

# No unnamed columns so, we won't change anything :)

# To check the data types & change it if we need:
data.dtypes

#To change the type of the column we may need:
data['first_open'] = pd.to_datetime(data['first_open'])
data['hour'] = pd.to_timedelta(data['hour'])
data['enrolled_date'] = pd.to_datetime(data['enrolled_date'])
data.dtypes

# To check the null values & drop or filling them if we need:
data.isnull().sum()

# we found null values in enrolled_date column so we will drop it cause we don't have the info to fill them:
data.dropna(inplace = True)
data

# To check the duplicates and drop them:
data.duplicated().sum()

# we found duplicated values so, we will drop it: 
data.drop_duplicates(inplace = True)
data

# The K_Medoids Algorithm: 
sample = data.loc[0:5000]
columns_var = sample[['age','user']]
data_array = np.array(columns_var)
k = 2
counter0 = 0
counter1 = 0
k_mediods = KMedoids(n_clusters = k).fit(data_array)
clusters = k_mediods.cluster_centers_
labels = k_mediods.labels_

for i in range(k):
    print("[[{:0.1f}, {:d}]]".format(clusters[i][0], int(clusters[i][1])))

print(labels)

for j in range(k) : 
    for i in range(len(data_array)):    
        if k_mediods.labels_[i] == j:
            x = data_array[i]
            print('Cluster ' , j, ':', x)
            if j == 0:
                counter0 += 1

            elif j == 1:
                counter1 += 1        
                
                
# The hirerachial Method: 
# Slicing the data
sub_sample = data.loc[0:50]
columns_sub_sample = sub_sample[['age','user']]
data_array_sub_sample = np.array(columns_sub_sample)

# Graghing the data
plt.figure(figsize=(10, 6))
plt.scatter(data_array_sub_sample[:, 0], data_array_sub_sample[:, 1], c = 'r')  
plt.xlabel('age')
plt.ylabel('user')
plt.grid()
plt.show()

# The linkage functions
z1 = linkage(data_array, method = 'single', metric = 'euclidean')
z2 = linkage(data_array, method = 'complete', metric = 'euclidean')
z3 = linkage(data_array, method = 'average', metric = 'euclidean')
z4 = linkage(data_array, method = 'ward', metric = 'euclidean')

# The dendrogram 
plt.figure(figsize = (15, 20))

plt.subplot(2,2,1), dendrogram(z1, truncate_mode = 'lastp', p = 10), plt.title('Single')
plt.subplot(2,2,2), dendrogram(z2, truncate_mode = 'lastp', p = 10), plt.title('Complete')
plt.subplot(2,2,3), dendrogram(z3, truncate_mode = 'lastp', p = 10), plt.title('Average')
plt.subplot(2,2,4), dendrogram(z4, truncate_mode = 'lastp', p = 10), plt.title('Ward')

plt.show()

#Distribution of users login by Day of the Week
weekly_counts = data['dayofweek'].value_counts().sort_index()
weekly_counts=pd.DataFrame({'dayofweek':weekly_counts.index,'count':weekly_counts.values})
day_labels = ['Sat', 'Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri']
fig = px.pie(values=weekly_counts['count'],
             names=day_labels,
             title='Distribution of users login by Day of the Week',
             hole=0.7)
fig.show()

#relation between age and number of screens
bar_color = 'blue'
fig = px.bar(x=data['age'], y=data['numscreens'], 
             title="Age vs Number of Screens Used",
             color_discrete_sequence=[bar_color])
fig.update_xaxes(title_text="Age")
fig.update_yaxes(title_text="Number of Screens")
fig.show()

#number of users over years
data['year'] = data['first_open'].dt.year
user_counts_per_year = data.groupby('year')['user'].nunique()
user_counts_df = pd.DataFrame({'year': user_counts_per_year.index, 'user_count': user_counts_per_year.values})
fig = px.area(user_counts_df, x='year', y='user_count', 
              title='Total Number of Users Over the Years',
              labels={'year': 'Year', 'user_count': 'Number of Users'})
fig.show()

#Number of users who liked the app compared to those who did not
like = data['liked'].value_counts().sort_index()
like_df = pd.DataFrame({'liked': like.index, 'count': like.values})
fig=px.bar(x=like_df['liked'],y=like_df['count'],title='Users Who Liked the App vs. Users Who Did Not')
fig.update_xaxes(title_text='liked')
fig.update_yaxes(title_text='count')
fig.update_layout(width=1100, height=600)
fig.show()


