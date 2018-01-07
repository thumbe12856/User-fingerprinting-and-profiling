import pandas as pd 
from IPython.display import display,clear_output
import numpy as np
from sklearn import tree

trainingFile = 'traing_data2.csv'
testingFile = 'testing_data2.csv'

print 'preprocessing start'
dataset = pd.read_csv(trainingFile)
dataset['IP label'] = dataset.groupby('IP address').grouper.group_info[0] + 1

dataset.to_csv('training_data2_with_label.csv', index=False)



#建立IP label dataframe
IP_label = dataset[['IP label', 'IP address']].copy()
IP_label = IP_label.drop_duplicates(['IP label'])
IP_label = IP_label.drop_duplicates(['IP address'])
IP_label = IP_label.reset_index(drop=True)

test = pd.read_csv('testing_data2.csv')
test_Y = test.drop(['Domain', 'flag', 'date'], 1)
test_Y = test_Y.values

#本來想做LABEL 檔
label=[]
#for i in range(test_Y.shape[0]):
for i in range(len(test_Y)):
    for j in range( IP_label.shape[0]):
        if  IP_label.iloc[j, 1] == test_Y[i]:
            label.append(IP_label.iloc[j, 0])
        else :
            label.append(-1)
    print i

a = np.array( label )
df = pd.DataFrame(a)
df.to_csv('testing_data2_with_label.csv',index=False)
print 'finish'


# In[ ]:



