import csv
import math
from random import randint
def readCSV(filename):
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        groupList = list(reader)
        listHead = groupList[1]
        groupList = groupList[2:len(groupList)]
    return groupList, listHead

# calculate euclidean distance of two lists
def euclidean(list0, list1, weight):
    distance = 0
    # totalWeight = 0
    # for i in range(len(weight)):
    #     totalWeight += weight[i]
    for i in range(len(list0)):
        # distance += math.fabs((float(list0[i]) - float(list1[i]))) * float(weight[i])
        distance += math.pow((float(list0[i]) - float(list1[i])), 2) * float(weight[i])
    return math.sqrt(abs(distance))

# calculate similar pair in a group of list (each group has three cases)
def dist(list0, list1, list2, weight):
    length = len(list0)
    list0 = list0[1:length]
    list1 = list1[1:length]
    list2 = list2[1:length]
    dist = []
    dist.append(euclidean(list0, list1, weight))
    dist.append(euclidean(list1, list2, weight))
    dist.append(euclidean(list0, list2, weight))
    if dist.index(min(dist)) == 0:
        return "01"
    elif dist.index(min(dist)) == 1:
        return "12"
    else:
        return "02"

def calAccuracy(pairs1, pairs2):
    if len(pairs1) != len(pairs2):
        print "Pairs length not match!"
    total = len(pairs1)
    count = 0
    for i in range(total):
        if pairs1[i] == pairs2[i]:
            count += 1
#         else:
#             print "Do not match: Pair"+str(i)

    return float(count) / float(total)

def calSimilarPairs(groupList, weight):
    similarPairs = []
    for i in range(len(groupList)/3):
        list0 = groupList[i]
        list1 = groupList[i + 1]
        list2 = groupList[i + 2]
        similarPairs.append(dist(list0, list1, list2, weight))
    return similarPairs

groupList, listHead = readCSV("CopyofRandGroup.csv")
    # print groupList
coloums = len(groupList[0])
rows = len(groupList)
    #
    # Uniform each coloum
    #
maxAttr = 0
for i in range(coloums):
    if i > 0:
        for j in range(rows):
            if float(groupList[j][i]) > float(maxAttr):
                maxAttr = float(groupList[j][i])
        for j in range(rows):
            groupList[j][i] = float(float(groupList[j][i]) / maxAttr)
        maxAttr = 0
# print groupList
    #
    # weight = 1, calculate accuracy
    #
weight = []
for i in range(len(groupList[0]) - 1):
    weight.append(1)
similarPairs = calSimilarPairs(groupList, weight)
    #
    # Similar pairs of patients in each group labeled by doctor
    #
similarPairs_Dr = ['12','02','12','02','12','02','01','01','12','02','02','01','01','02','01','12','01','12','01','01','02','12','12','12','12','12','01','12','12','12','01','12','02','02','12','12','02','12','01','12','12']
accuracy = calAccuracy(similarPairs, similarPairs_Dr)
print "weight = 1: Accuracy " +str(accuracy)     

#######brute-force algorithm
max_weight = []
for i in range(0,len(weight)):
    max_accu = 0
    max_weight = 0
    for item in range(0,101):
        weight[i] = item
        # print weight
        similarPairs_2= calSimilarPairs(groupList, weight)
        accuracy = calAccuracy(similarPairs_2, similarPairs_Dr) 
        # print accuracy
        if accuracy > max_accu:
            max_accu = accuracy
            max_weight = item
        # print max_weight
    weight[i] = max_weight 

print "Now weight"
print weight
print "Now accuracy"
print max_accu

#######new method
similarPairs_Dr = ['12','02','12','02','12','02','01','01','12','02','02','01','01','02','01','12','01','12','01','01','02','12','12','12','12','12','01','12','12','12','01','12','02','02','12','12','02','12','01','12','12']
similarPairs_list = []
for item in similarPairs_Dr:
	temp =[]
	temp.append(int(item)/10)
	temp.append(int(item)%10)
	similarPairs_list.append(temp)
weight = []
for i in range(len(groupList[0]) - 1):
	weight.append(1)
max_accu = 0
max_a = 0
max_weight_list = []
a_list =[]
b_list= []
for a in range(1,5):
	for b in range(1,5):
		for i in range(0,len(similarPairs_Dr)):
			select = i*3 + similarPairs_list[i][0]
			pos = i*3 + similarPairs_list[i][1]
			neg = i*3 + 3 - select - pos
			for j in range(0,len(weight)):
				weight[j] = weight[j] + a*abs(int(groupList[select][j+1])-int(groupList[neg][j+1])) - b*abs(int(groupList[select][j+1])-int(groupList[pos][j+1]))
				similarPairs_2= calSimilarPairs(groupList, weight)
					#             print weight
				accuracy = calAccuracy(similarPairs_2, similarPairs_Dr)
						#             print accuracy
				if accuracy > max_accu:
					max_accu = accuracy
					max_weight_list.append(weight)
					max_a = a
					max_b = b

print "Now accuracy"
print max_accu
print "Now weight"
print max_weight_list[-1]
