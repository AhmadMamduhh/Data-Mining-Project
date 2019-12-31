import numpy as np
import pandas as pd
from visualizer import Visualizer

# ------------Loading the data-------------------

# 1 = classification, 2 = regression and any other number = clustering
method_identifier = int(input('Enter 1 to choose Classification, 2 to choose Regression' +
                              ' or 3 to choose Clustering: '))

if method_identifier == 1:
    dataset = pd.read_csv('wisconsin_breast_cancer.csv')
elif method_identifier == 2:
    dataset = pd.read_csv('diamonds.csv')
elif method_identifier == 3:
    from sklearn.datasets import load_iris

    X = load_iris(return_X_y=False)['data']
    iris = load_iris()

# diamonds.csv on kaggle.com for regression
# wisonsin breast cancer.cvs on kaggle.com for classification
# iris.csv for clustering (drop class column)


# --------------Choosing the desired algorithm--------------------

# Choosing classification algorithm
if method_identifier == 1:
    identifier = int(input('Enter 1 to choose KNN, 2 to choose Decision Tree, 3 to choose Naive Bayes' +
                           ', 4 to choose Random Forest or 5 to choose Neural Network: '))
    if identifier == 1:
        algorithm_name = 'KNN'
    elif identifier == 2:
        algorithm_name = 'Decision Tree'
    elif identifier == 3:
        algorithm_name = 'Naive Bayes'
    elif identifier == 4:
        algorithm_name = 'Random Forest'
    else:
        algorithm_name = 'Neural Network'

# Choosing Regression algorithm
elif method_identifier == 2:
    identifier = int(input('Enter 1 to choose Linear Regression, 2 to choose Polynomial Regression,' +
                           ' 3 to choose Decision Tree, 4 to choose KNN Regression, 5 to choose' +
                           ' Random Forest Regressor or 6 to choose Neural Network: '))
    if identifier == 1:
        algorithm_name = 'Linear Regression'
    elif identifier == 2:
        algorithm_name = 'Polynomial Regression'
    elif identifier == 3:
        algorithm_name = 'Decision Tree'
    elif identifier == 4:
        algorithm_name = 'KNN Regression'
    elif identifier == 5:
        algorithm_name = 'Random Forest'
    else:
        algorithm_name = 'Neural Network'

# Choosing Clustering algorithm
else:
    algorithm_name = 'K-Means'

# ---------------------Preprocessing the data---------------------------------
from preprocessor import Preprocessor

preprocess = Preprocessor()

# Cleaning the data
if method_identifier == 3: # Drop missing rows in iris dataset
    X = preprocess.drop_missing(X)
else: # Drop missing rows in diamond or wisconsin breast cancer datasets
    dataset = preprocess.drop_missing(dataset)

if method_identifier == 1:
    X, y = preprocess.dataframe_to_numpy(dataset, 'breast cancer')
    X, y = preprocess.encoding(X, y, 'breast cancer')

elif method_identifier == 2:
    X, y = preprocess.dataframe_to_numpy(dataset, 'diamonds')
    X, y = preprocess.encoding(X, y, 'diamonds')

# Splitting the data into train and test sets
if method_identifier == 1 or method_identifier == 2:
    X_train, X_test, y_train, y_test = preprocess.split_data(X, 0.19, y)

elif method_identifier == 3:
    X_train, X_test = preprocess.split_data(X, test_ratio=0.3)

# Scaling the data
X_train, X_test = preprocess.scaling(X_train, X_test, scale_type='Standard Scaler')

# ----------------------------Classifying the data----------------------------
if method_identifier == 1:

    from classifier import Classifier
    
    classifier = Classifier(algorithm_name)
    y_predicted = classifier.classify(X_train, y_train, X_test, y_test)
    classifier_accuracy = classifier.get_accuracy(y_test, y_predicted)

    # Visualizing the results
    visualizer = Visualizer()
    visualizer.plot_classifier_regressor(y_test, y_predicted, method_identifier)
    
    print('The accuracy is: ' + str(classifier_accuracy) + ' %')
    print(algorithm_name)

# ---------------------Applying Regression to the data--------------------------
elif method_identifier == 2:

    from regressor import Regressor

    regressor = Regressor(algorithm_name)
    y_predicted = regressor.predict(X_train, y_train, X_test)
    regressor_score = regressor.get_score(y_test, y_predicted)

    # Visualizing the results
    visualizer = Visualizer()
    visualizer.plot_classifier_regressor(y_test, y_predicted, method_identifier)

    print('The coefficient of determination is: ' + str(regressor_score))
    print(algorithm_name)

# ---------------------Clustering the data------------------------------------
elif method_identifier == 3:

    from clustering import Clustering

    clustering = Clustering(algorithm_name)
    n_clusters, inertia = clustering.tune_parameters(X_train)
    clusters = clustering.cluster(X_train, X_test, n_clusters)

    # Visualizing the results
    visualizer = Visualizer()
    visualizer.plot_clustering(X_test, clusters)

    print("The clustering model's inertia: " + str(inertia))
    print(str(algorithm_name))
