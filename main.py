import numpy as np
import pandas as pd

# Load the data
from sklearn.datasets import load_boston, load_iris
X , y = load_iris(return_X_y = True)

# Splitting the data into train and test sets
from preprocessor import Preprocessor
preprocess = Preprocessor(X , y)
X_train, X_test, y_train, y_test = preprocess.split_data(test_size = 0.33)

# Choosing classifier name

identifier = int(input('Enter 1 to choose KNN, or 2 to choose decision tree: '))
print(str(identifier))
if(identifier == 1):
    algorithm_name = 'KNN'
elif(identifier == 2):
    algorithm_name = 'Decision Tree'
    
    
# Classify the data

from classifier import Classifier
    
classifier = Classifier(algorithm_name , X_train , y_train)
y_predicted = classifier.classify(X_test)
print(y_predicted)
print(y_test)
print('The accuracy is: ' + str(int((np.sum(y_test == y_predicted) / y_test.shape[0])*100)) + '%')
print(algorithm_name)
