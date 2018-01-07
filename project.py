
# coding: utf-8

# In[25]:

import pandas as pd 
from IPython.display import display

column_names = ['IP address','timestamp','Domain','flag']
dataset = pd.read_csv('dataset.txt', delim_whitespace=True,names = column_names)
dataset['date'] = pd.to_datetime(dataset['timestamp'],unit='s')
dataset = dataset.drop('timestamp', 1)
mask1 = (dataset['date'] >= '2017-01-06 00:00:00') & (dataset['date'] <='2017-01-06 23:59:59')
mask2 = (dataset['date'] >= '2017-02-01 00:00:00') & (dataset['date'] <='2017-02-16 23:59:59')
mask3 = (dataset['date'] == '2017-01-06 00:00:00')
dataset1 = dataset.loc[mask1]
dataset2 = dataset.loc[mask2]
dataset3 = dataset.loc[mask3]
# display(dataset1)
# display(dataset2)
dataset1.to_csv('training_data1.csv', index=False)
dataset2.to_csv('testing_data1.csv', index=False)
# # print dataset.groupby('Domain').count()
# for name, group in dataset1.groupby('IP address'):
#     file_name=str(name)+'.csv'
#     group.to_csv(file_name, index=False)


# In[31]:

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


dataset1['IP label'] = dataset1.groupby('IP address').grouper.group_info[0] + 1
display(dataset1)
# for name, group in dataset1.groupby('IP address'):
#     file_name=str(name)+'.csv'
#     group.to_csv(file_name, index=False)


# In[ ]:

from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


task_1 = dataset1
estimator1 = KMeans(n_clusters=1000).fit_predict(task_1)
plt.scatter(task_1.IP_address, task_1.Domain, c=estimator1)
plt.title('K Means')
plt.show()


# In[ ]:



