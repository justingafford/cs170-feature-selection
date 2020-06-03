from pprint import pprint
import math, random, heapq, copy

# here we populate our heap based on their accuracy
def PopulateHeap(data, instanceCount, featureCount, allFeatures):
    TheQueue = []
    
    for i in range(1, featureCount + 1):
        if (i not in allFeatures):
            accuracy = CrossValidation(data, instanceCount, \
                allFeatures, i)
            featureMatch = (1 - accuracy, i)
            heapq.heappush(TheQueue, featureMatch)

    return TheQueue

# we use this algorithm in order to add a feature to the heap
def AddFeatureToHeap(data, instanceCount, featureCount, allFeatures, TheQueue):
    first = heapq.heappop(TheQueue)
    newFeature = first[1]
    accuracy = CrossValidation(data, instanceCount, \ allFeatures, newFeature)
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
    
def NearestNeighbor(instances, instanceCount, removeOne, featuresPrint):

def Classification(instances, nn, removeOne):

def main():
    
if __name__ == "__main__":
    main()
 
