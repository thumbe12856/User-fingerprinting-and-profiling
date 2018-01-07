def labeling(threadId, fromIndex, toIndex, test_YY):
    print('threadId: %d, fromIndex: %d, toIndex: %d' % ( threadId, fromIndex, toIndex ))
    print('start')
    for j in range(len(IP_label)):
        if  IP_label.iloc[j, 1] == test_YY[i]:
            label.append(IP_label.iloc[j, 0])
        else :
            label.append(-1)
    print('threadId: %d, fromIndex: %d, toIndex: %d' % ( threadId, fromIndex, toIndex ))
    print('done')

totalSize = len(test_Y)
test_Y0 = test_Y
test_Y1 = test_Y
test_Y2 = test_Y
test_Y3 = test_Y
test_Y4 = test_Y
test_Y5 = test_Y
test_Y6 = test_Y
test_Y7 = test_Y


threadSize = 8
temp = []
for i in range(threadSize):
	fromIndex = i * (totalSize / threadSize)
	toIndex =(i + 1) * (totalSize / threadSize)
	if(i == threadSize - 1):
		toIndex = totalSize
	if(i == 0):
		thread = threading.Thread(target = labeling, args=(i, fromIndex, toIndex, test_Y0))
	elif(i == 1):
		thread = threading.Thread(target = labeling, args=(i, fromIndex, toIndex, test_Y1))
	elif(i == 2):
		thread = threading.Thread(target = labeling, args=(i, fromIndex, toIndex, test_Y2))
	elif(i == 3):
		thread = threading.Thread(target = labeling, args=(i, fromIndex, toIndex, test_Y3))
	elif(i == 4):
		thread = threading.Thread(target = labeling, args=(i, fromIndex, toIndex, test_Y4))
	elif(i == 5):
		thread = threading.Thread(target = labeling, args=(i, fromIndex, toIndex, test_Y5))
	elif(i == 6):
		thread = threading.Thread(target = labeling, args=(i, fromIndex, toIndex, test_Y6))
	elif(i == 7):
		thread = threading.Thread(target = labeling, args=(i, fromIndex, toIndex, test_Y7))
	thread.start()
	temp.append(thread)

for i in range(threadSize):
    temp[i].join()

