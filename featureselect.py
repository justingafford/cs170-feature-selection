from pprint import pprint
import math, random, heapq, copy

# here we populate our heap based on their accuracy
def PopulateHeap(data, instanceCount, featureCount, allFeatures):
    TheQueue = []
    
    for i in range(1, featureCount + 1):
        if (i not in allFeatures):
            accuracy = CrossValidation(data, instanceCount, allFeatures, i)
            featureMatch = (1 - accuracy, i)
            heapq.heappush(TheQueue, featureMatch)

    return TheQueue

# we use this algorithm in order to add a feature to the heap
def AddFeatureToHeap(data, instanceCount, featureCount, allFeatures, TheQueue):
    first = heapq.heappop(TheQueue)
    newFeature = first[1]
    accuracy = CrossValidation(data, instanceCount, allFeatures, newFeature)
    return newFeature, accuracy

# uses one out k-fold cross validation in order to predict the accuracy of our data
#we use pos. numbers to add, neg. to remove, 0 when there's no features(One out)
def CrossValidation(instances, instanceCount, existingFeatures, custFeature):
    if custFeature > 0:
        featuresPrint = list(existingFeatures)
        featuresPrint.append(custFeature)
    elif custFeature < 0:
        custFeature = custFeature * -1
        existingFeatures.remove(custFeature)
        featuresPrint = list(existingFeatures)
        existingFeatures.add(custFeature)
    elif custFeature == 0:
        featuresPrint = list(existingFeatures)
    correct = 0
    
    for i in range(0, instanceCount):
        removeOne = i
        nn = NearestNeighbor(instances, instanceCount, removeOne, featuresPrint)
        correctClass = Classification(instances, nn, removeOne)
        if (correctClass):
            correct += 1

    accuracy = correct / instanceCount
    print("Testing features: ", featuresPrint, " with accuracy %f" % accuracy)
    return accuracy
#A basic nearest neighbor calcualation in which we calcualate the nearest neighbor of features    
def NearestNeighbor(instances, instanceCount, removeOne, features):
    nn = -1
    nnDist = float('inf')
    featureCount = len(features)
    
    for i in range(0, instanceCount):
        if (i == removeOne):
            pass
        else:
            sum = 0
            
            for j in range(0, featureCount):
                sum = sum + pow((instances[i][features[j]] - instances[removeOne][features[j]]), 2)
                
            distance = math.sqrt(sum)
            if distance < nnDist:
                nnDist = distance
                nn = i
                
    return nn

#function where we check the classification based upon the nearest neighbor
def Classification(instances, nn, removeOne):
    if (instances[nn][0] != instances[removeOne][0]):
        return False
    return True

#here we read in the data from the input file(database file) and return the instances
def ReadData(filename, instanceCount):
    try:
        file = open(filename, 'r')
    except:
        raise FileNotFoundError(filename)
    instances = [[] for i in range(instanceCount)]
    
    for i in range(instanceCount):
        instances[i] = [float(j) for j in file.readline().split()]
        
    return instances
#here we normalize the data given from the text file, normInstances is temp value(makes sure we dont update instances
def NormData(instances, instanceCount, featureCount):
    normInstances = list(instances)
    mean = CalcMean(instances, instanceCount, featureCount)
    std = CalcStd(instances, instanceCount, featureCount, mean)
    
    for i in range(0, instanceCount):
        for j in range(1, featureCount + 1):
            normInstances[i][j] = ((instances[i][j] - mean[j - 1]) / std[j - 1])

    return normInstances

#calculates the average used in our normdata def
def CalcMean(instances, instanceCount, featureCount):
    mean = []
    
    for i in range(1, featureCount + 1): # Add one to exclude the class data
        mean.append((sum(row[i] for row in instances)) / instanceCount)
        
    return mean

#calcualtes the standard deviation for our normdata def
def CalcStd(instances, instanceCount, featureCount, mean):
    std = []
    
    for i in range(1, featureCount + 1):
        std.append(math.sqrt((sum(pow((row[i] - mean[i - 1]), 2) for row in instances)) / instanceCount))
        
    return std

def main():
    print("Welcome to Justin Gaffords Feature Selection algorithm")
    filename = input("Type in the name of the file to test:")
    instanceCount = int(input("\nEnter the number of instances to read:"))
    instances = ReadData(filename, instanceCount)
    featureCount = len(instances[0]) - 1
    print("\nThis dataset has", featureCount, "features, with",\
        instanceCount, "instances")
    print("\nPlease wait while I normalize the data... Done!")
    normInstances = NormData(instances, instanceCount, featureCount)
    algorithmChoice = ""
    
    while (algorithmChoice != "1" and algorithmChoice != "2" and algorithmChoice != "3"):
        algorithmChoice = input("""\nType in the number of the algorithm you want to run:\n
                       1) Forward Selection
                       2) Backward Elimination
                       3) Custom Search
                    \r""")
    
    if (algorithmChoice == "1"):
        ForwardSelection(normInstances, instanceCount, featureCount)
    elif (algorithmChoice == "2"):
        BackwardElimination(normInstances, instanceCount, featureCount)
    else:
        CustomSearch(normInstances, instanceCount, featureCount)
    
if __name__ == "__main__":
    main()
 
