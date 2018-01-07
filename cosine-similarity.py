import numpy as np
import pandas as pd 
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from pathlib import Path

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


""" main """
if __name__ == "__main__":

	correctMean = 0
	allFiles = 4
	dataset1Len = 0
	dataset2Len = 0
	for i in range(allFiles):
		trainingFile = './data/training/6_hour/training_data2_6hour_' + str(i * 6) + '.csv'
		testingFile = './data/testing/6_hour/testing_data2_6hour_' + str(i * 6) +'.csv'

		trainingFileExist = Path(trainingFile)
		testingFileExist = Path(testingFile)
		if(trainingFileExist.exists() & testingFileExist.exists()):

			""" reaed testing  and training file """
			column_names = ['IP', 'Domain', 'flag', 'date']
			dataset1 = pd.read_csv(trainingFile)
			dataset2 = pd.read_csv(testingFile)
			dataset1Len += len(dataset1)
			dataset2Len += len(dataset2)
			dataset1.columns = column_names
			dataset2.columns = column_names
			print("1. Read files %s finish." % trainingFile)

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
			print("2.1 Transform trainging data to X finish.")


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
			print("2.2 Transform testing data to Y finish.")


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
					weight = 1
				domainWeight[i] =  weight
				i = i + 1
			print("3. Calculate domains' weight finish.")


			''' for toy test '''
			# trainingX = realTrainX[:5].copy()
			# testingY = realTestY[:5].copy()


			''' calculating accuracy '''
			correct = 0
			nowIndex = 0

			for testDataIP in realTestY.index:
				# print nowIndex, '->', len(realTestY.index)
				nowIndex = nowIndex + 1
				b = realTestY.loc[testDataIP].values
				maxSum = -1
				maxSumIP = -1
				tempSum = 0
				for trainDataIP in realTrainX.index:
					a = realTrainX.loc[trainDataIP].values
					''' do vector inner product '''
					tempSum = (a * b * domainWeight).sum()
					if(tempSum > maxSum):
						maxSum = tempSum
						maxSumIP = trainDataIP
				if(testDataIP == maxSumIP):
					correct = correct + 1

			print("4. Calculate predicting accuracy finish.")
			print "accuracy:", correct / float(len(realTestY.index))
			correctMean += correct / float(len(realTestY.index))
			print(correctMean / float(allFiles)), "\n"

	print("5. all file finish, mean accuracy:")
	print(correctMean / float(allFiles))
	print(dataset1Len / float(allFiles))
	print(dataset2Len / float(allFiles))
