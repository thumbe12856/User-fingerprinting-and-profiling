import pandas as pd 
from IPython.display import display

column_names = ['IP address','timestamp','Domain','flag']
dataset = pd.read_csv('dataset.txt', delim_whitespace=True,names = column_names)
dataset['date'] = pd.to_datetime(dataset['timestamp'],unit='s')
dataset = dataset.drop('timestamp', 1)
mask1 = (dataset['date'] >= '2017-01-15 00:00:00') & (dataset['date'] <='2017-01-31 23:59:59')
mask2 = (dataset['date'] >= '2017-02-01 00:00:00') & (dataset['date'] <='2017-02-16 23:59:59')
dataset1 = dataset.loc[mask1]
dataset2 = dataset.loc[mask2]
# display(dataset1)
# display(dataset2)
# dataset1.to_csv('training_data1.csv', index=False)
# dataset2.to_csv('testing_data1.csv', index=False)
# # print dataset.groupby('Domain').count()


# In[41]:

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)


def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
    
dataset1['IP label'] = dataset1.groupby('IP address').grouper.group_info[0] + 1
dataset1['IP address'] = dataset1['IP address'].astype('str')

X = dataset1.groupby(['IP address','Domain'])['IP label'].sum().unstack().reset_index().fillna(0).set_index('IP address')
X_sets = X.applymap(encode_units)
display(X_sets)


# In[44]:




# In[46]:



frequent_itemsets = apriori(X_sets, min_support=0.7, use_colnames=True)



# In[47]:

display(frequent_itemsets)


# In[ ]:



