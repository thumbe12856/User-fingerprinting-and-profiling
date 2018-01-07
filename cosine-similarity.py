import numpy as np
import pandas as pd 
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def getTTvalue(allTrainingDataset, allTestingDataset):
	returnData = pd.DataFrame()
	for trainDataIP in allTrainingDataset['IP'].index:
		dataset3 = allTestingDataset.copy()
		as_list = dataset3.index.tolist()
		if(trainDataIP in as_list):
			idx = as_list.index(trainDataIP)
			as_list[idx] = 'South Korea'
			dataset3.index = as_list
		allData = allTrainingDataset.append(dataset3).fillna(0)
		returnData = returnData.append(allData[allData.index == trainDataIP])
	return returnData

def mutiplyVector(a, b, weight):
	temp = 0
	for i in range(a.size):
		temp += a[i] * b[i] * weight
	return temp


trainingFile = 'traing_data2.csv'
testingFile = 'testing_data2.csv'

column_names = ['IP', 'Domain', 'flag', 'date']
dataset1 = pd.read_csv(trainingFile)
dataset2 = pd.read_csv(testingFile)
dataset1.columns = column_names
dataset2.columns = column_names

"""

	allDataset:
	+-------------------+
	|	training data 	|
	+-------------------+
	|	testing  data 	|
	+-------------------+

"""
allDataset = dataset1.append(dataset2)


allTrainingDataset = pd.DataFrame({'count': dataset1.groupby(['IP', 'Domain']).size()}).unstack().reset_index().fillna(0).set_index('IP')
allTestingDataset = pd.DataFrame({'count': dataset2.groupby(['IP', 'Domain']).size()}).unstack().reset_index().fillna(0).set_index('IP')


"""

	realTrainX: when in training data time
	+---------------------------------------------------------------------------------+
	|      | visit domain 1 times | visit domain 2 times | ... | visit domain N times |
	|---------------------------------------------------------------------------------|
	| IP 1 |                      |                      |     |                      |
	|---------------------------------------------------------------------------------|
	| IP 2 |                      |                      |     |                      |
	|---------------------------------------------------------------------------------|
	|                                        .                                        |	
	|                                        .                                        |
	|                                        .                                        |
	|---------------------------------------------------------------------------------|
	| IP N |                      |                      |     |                      |
	+---------------------------------------------------------------------------------+

"""
realTrainX = getTTvalue(allTrainingDataset, allTestingDataset)


"""

	realTestY: when in testing data time
	+---------------------------------------------------------------------------------+
	|      | visit domain 1 times | visit domain 2 times | ... | visit domain N times |
	|---------------------------------------------------------------------------------|
	| IP 1 |                      |                      |     |                      |
	|---------------------------------------------------------------------------------|
	| IP 2 |                      |                      |     |                      |
	|---------------------------------------------------------------------------------|
	|                                        .                                        |	
	|                                        .                                        |
	|                                        .                                        |
	|---------------------------------------------------------------------------------|
	| IP N |                      |                      |     |                      |
	+---------------------------------------------------------------------------------+

"""
realTestY = getTTvalue(allTestingDataset, allTrainingDataset)


"""

	domainFlag: (type: pandas dataFrame)
	+----------------------------+
	|		   | is core domain? |
	|----------------------------|
	| damain 1 |                 |
	|----------------------------|
	| damain 2 |                 |
	|----------------------------|
	|             .              |
	|             .              |
	|             .              |
	|----------------------------|
	| damain N |                 |
	+----------------------------+

"""
domainFlag = allDataset.drop(['IP', 'date'])
domainFlag = domainFlag.drop_duplicates(subset=['Domain'], keep='first')
domainFlag = domainFlag.reset_index(drop=True)
domainFlag = domainFlag.set_index('Domain')


"""
	domainWeight: (type: numpy array)
	+-------------------------------------------------------+
	|                | domain 1 | domain 2 | ... | domain N |
	|-------------------------------------------------------|
	| is core domain |          |          | ... |          |
	+-------------------------------------------------------+

"""
domainWeight = np.array([0] * len(realTrainX['count'].columns))
i = 0
for domainID in realTrainX['count'].columns:
	flag = domainFlag.loc[domainID]['flag']
	if(flag):
		weight = 1
	else:
		weight = 0.1
	domainWeight[i] =  weight
	i = i + 1

trainingX = realTrainX[:5].copy()
testingY = realTestY[:5].copy()



correct = 0
nowIndex = 0

''' do vector inner product '''
for testDataIP in realTestY.index:
	print nowIndex, '->', len(realTestY.index)
	nowIndex = nowIndex + 1
	b = realTestY.loc[testDataIP].values
	maxSum = -1
	maxSumIP = -1
	tempSum = 0
	for trainDataIP in realTrainX.index:
		a = realTrainX.loc[trainDataIP].values
		tempSum = (a * b * domainWeight).sum()
		#tempSum = (a * b).sum()
		if(tempSum > maxSum):
			maxSum = tempSum
			maxSumIP = trainDataIP
	if(testDataIP == maxSumIP):
		correct = correct + 1

print correct


