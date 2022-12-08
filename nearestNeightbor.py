import numpy as np
"""
Class: nearestNeighbor
    This class serves to hold the necessary functions to implement
    nearest neighbor classification, accuracy of datasets using leave one out,
    Forward / Backwards search, and necessary helper functions.
"""

"""
nearestNeighbor()
instanceRow: (int) Index of the row to find the nearest neighbor to
dataSet: 2D Array of the full data set
"""
def nearestNeighbor(instanceRow, dataSet):
    # Get row from dataset of instance
    instance = dataSet[instanceRow]
    # Convert row to a numpy array for faster processing
    instance = np.array(instance)
    # print("instance:", instance)
    # Class of instance - First element in array
    instanceClass = instance[0]

    # Set distance and row/location of nearestNeighbor to infinity
    nearestNeighborDistance = float('inf')
    nearestNeighborInstance = None
    # Iterate through dataset and calculate nearest neighbor
    for k in range(len(dataSet)):
        # Get kth row from dataSet
        row = dataSet[k]
        # Convert dataSet row to numpy array for faster processing
        row = np.array(row)
        if k != instanceRow:
            # Use Euclidean Distance formula
            distance = (np.subtract(instance[1:], row[1:]))
            distance = np.power(distance, 2)
            distance = sum(distance)
            distance = np.sqrt(distance)
            if distance < nearestNeighborDistance:
                nearestNeighborDistance = distance
                nearestNeighborInstance = row
    return nearestNeighborInstance

"""
accuracy()
dataSet: 2D Array of the full dataset to test
This function finds the nearest neighbor for each element in the
dataSet, and checks if that nearestNeighbor is actually what the
instance is. Essentially, leave one out cross validation.
"""
def accuracy(dataSet):
    DEBUG = False
    if DEBUG:
        print("Dataset: ")
        print(dataSet)
    correctlyClassifiedNum = 0
    for i in range(len(dataSet)):
        # print("=====")
        nn = nearestNeighbor(i, dataSet)
        if dataSet[i][0] == nn[0]:
            # print("Correct Neighbor! ")
            correctlyClassifiedNum += 1
    # print("Accuracy Function:", correctlyClassifiedNum / len(dataSet))
    return correctlyClassifiedNum / len(dataSet)

"""
forwardSelection()
dataSet: 2D Array of data
This function is a search algorithm that finds the most optimal set 
of features 
"""
def forwardSelection(dataSet):
    rowLength = len(dataSet[0])
    numFeatures = rowLength - 1
    currentSetFeatures = set()
    allTimeAccuracy = 0
    allTimeFeatures = set()
    for i in range(rowLength - 1):
        print("On the", i, "th level of the search tree")
        bestAccuracySoFar = 0
        bestFeatureSetSoFar = set()
        featureToAddThisLevel = False
        for k in range(1, rowLength):
            featureK = {k}
            if not currentSetFeatures.intersection(featureK):
                # print("Current Set Features", currentSetFeatures)
                print("--Considering adding the", k, "feature")
                currentAccuracy = leaveOneOutCrossValidation \
                    (dataSet, currentSetFeatures, k)
                print("Current Accuracy Found:", currentAccuracy)
                if currentAccuracy > bestAccuracySoFar:
                    print("Better accuracy found:", currentAccuracy)
                    bestAccuracySoFar = currentAccuracy
                    featureToAddThisLevel = k
        if featureToAddThisLevel:
            currentSetFeatures.add(featureToAddThisLevel)
        print("====================================")
        print("Best Accuracy:", bestAccuracySoFar)
        print("On level", i, "I added feature", featureToAddThisLevel)
        print("====================================")
        if bestAccuracySoFar > allTimeAccuracy:
            print("So Far:::", bestAccuracySoFar, "All Time:::", allTimeAccuracy)
            print(":::", currentSetFeatures)
            print("All Time Updated:", bestAccuracySoFar)
            allTimeAccuracy = bestAccuracySoFar
            allTimeFeatures = currentSetFeatures.copy()

    print("Best Ever:", allTimeFeatures, "Accuracy:", allTimeAccuracy)


"""
leaveOneOutCrossValidation()
dataSet: 2D Array of data
currentSetFeatures: set() data with current set of features
featureIndex: int() of candidate feature to add to currentSetFeatures
"""
def leaveOneOutCrossValidation(dataSet, currentSetFeatures, featureIndex):
    # Add feature to currentSet
    csf = currentSetFeatures.copy()
    csf.add(featureIndex)
    sorted(csf)
    # Now we have our dataset only with wanted features
    # Now, we execute nearestNeighbor and find accuracy
    return accuracy(createDataSet(dataSet, csf))


def leaveOneOutCVBackwards(dataSet, currentSetFeatures):
    csf = currentSetFeatures.copy()
    sorted(csf)
    return accuracy(createDataSet(dataSet, csf))

"""
createDataSet()
Parses a new 2D array only of currentSetFeatures
Helper Function to accuracy()
"""
def createDataSet(dataSet, currentSetFeatures):
    # print("Calculating CSF:", currentSetFeatures)
    # Make new data set with this new set of features
    # nearestNeighbor takes in a dataset iterated through fully
    featureDataSet = []
    for i in range(len(dataSet)):
        featureDataSet.append([dataSet[i][0]])
    for feature in currentSetFeatures:
        for i in range(len(dataSet)):
            featureDataSet[i].append(dataSet[i][feature])
    DEBUG = False
    if DEBUG:
        print("Feature Data Set - LOOV")
        print(featureDataSet)
    return featureDataSet

"""
backwardsElimination()
dataSet: 2D Array of data
"""
def backwardsElimination(dataSet):
    rowLength = len(dataSet[0])
    numFeatures = rowLength - 1

    # All Time
    allTimeAccuracy = 0
    allTimeFeatures = set()

    # Current Set of Features
    # Starts from full set
    currentSetFeatures = list(range(1, numFeatures + 1))

    for i in range(rowLength - 1, 0, -1):
        print("On the", i, "th level of the search tree")
        bestAccuracySoFar = 0
        bestFeatureSetSoFar = set()
        featureToRemoveThisLevel = False
        thisSet = None
        for k in range(numFeatures, 0, -1):
            thisSet = currentSetFeatures.copy()
            print("Current Set Features", currentSetFeatures)
            print("--Considering deleting the", thisSet[k - 1], "feature")
            del thisSet[k - 1]
            currentAccuracy = leaveOneOutCVBackwards(dataSet, thisSet)
            print("Current Accuracy Found:", currentAccuracy)
            if currentAccuracy > bestAccuracySoFar:
                print("Better Accuracy Found:", currentAccuracy)
                bestAccuracySoFar = currentAccuracy
                featureToRemoveThisLevel = k - 1
        numFeatures = numFeatures - 1
        if type(featureToRemoveThisLevel) == int:
            print("====================================")
            print("Best Accuracy:", bestAccuracySoFar)
            print("FTRT:", featureToRemoveThisLevel)
            print("On level", i, "I deleted feature", currentSetFeatures[featureToRemoveThisLevel])
            print("====================================")
            del currentSetFeatures[featureToRemoveThisLevel]

        if bestAccuracySoFar > allTimeAccuracy:
            print("So Far:::", bestAccuracySoFar, "All Time:::", allTimeAccuracy)
            print(":::", currentSetFeatures)
            print("All Time Updated:", bestAccuracySoFar)
            allTimeAccuracy = bestAccuracySoFar
            allTimeFeatures = currentSetFeatures.copy()

    print("Best Ever:", allTimeFeatures, "Accuracy:", allTimeAccuracy)
