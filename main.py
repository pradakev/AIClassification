import numpy as np
from nearestNeightbor import *

"""
Kevin Prada
CS 170: Artifical Intelligence
Professor EK - Fall 2022
Classification Algorithms
    Implement Nearest Neighbor Algorithms, then use Forward Search +
    Backwards Elimination to find best features to use for classes.
"""

# Program User Interface
print("Welcome to Kev's Feature Selection Algorithm.")

# Dataset Parsing
# fileName = input("Choose a file to parse: ")
fileName = "CS170_Large_Data__8.txt"
# fileName = "CS170_Small_Data__122.txt"

# Parse Dataset using loadfromtxt(), gathered from the numpy library
dataSet = np.loadtxt(fileName)

# Accuracy
accuracy1 = accuracy(dataSet) * 100

# Search Algorithm Selection
print('''Type in the number of the algorithm you'd like to run.''')
print("\t1) Forward Selection")
print("\t2) Backward Elimination")
choice = int(input())

# All Features Accuracy
numFeatures = len(dataSet[0]) - 1
print("This dataset has", numFeatures, "features (not including the class "
                                       "attribute).")
print("It contains", len(dataSet), "instances.")
print("Running nearest neighbor with all", numFeatures, "features, using leave "
        "one out evaluation, I get an accuracy of", accuracy1, "%")
print()
print("Beginning Search.")

# Search Algorithm Processing

if choice == 1:
    print("Forward Selection")
    forwardSelection(dataSet)
else:
    print("Backwards Elimination")
    backwardsElimination(dataSet)
