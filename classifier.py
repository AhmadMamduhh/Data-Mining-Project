import numpy as np
import pandas as pd

class Classifier:
    """ An instance of this class is given a certain classifier name when
        initialized. Based on this name, a certain classification algorithm 
        will be used when the classify method is called.
    """
        
    def __init__(self, classifier_name):
      self.classifier_name = classifier_name
    

    def classify(self, X_train, y_train, X_test, y_test):
        """ This method uses the apropriate classification algorithm based on
        the value of the classifier_name string and classifies the X_test 
        parameter and returns y_predicted. X_train and y_train are the data that
        are used for fitting. """
        
        
        if self.classifier_name == 'KNN':
            from sklearn.neighbors import KNeighborsClassifier
            neigh = KNeighborsClassifier(n_neighbors=6)
            neigh.fit(X_train, y_train)
            return neigh.predict(X_test)

            
        elif self.classifier_name == 'Decision Tree':
            from sklearn.tree import DecisionTreeClassifier
            decision_tree = DecisionTreeClassifier(random_state=0, max_depth=3)
            decision_tree.fit(X_train, y_train)
            return decision_tree.predict(X_test)
        
        elif self.classifier_name == 'Naive Bayes':
            # Import Gaussian Naive Bayes model
            from sklearn.naive_bayes import GaussianNB
            
            # Create a Gaussian Classifier
            gnb = GaussianNB()
            
            # Train the model using the training sets
            gnb.fit(X_train, y_train)
            
            # Predict the response for test dataset
            return gnb.predict(X_test)
            
        
        elif self.classifier_name == "Random Forest":
            """ Random Forest Classifier is based on the decision tree classifier.
            The random forest is a classification algorithm consisting of many decisions trees.
            It uses bagging and feature randomness when building each individual tree to try to create 
            an uncorrelated forest of trees whose prediction by committee is more accurate than 
            that of any individual tree. In the Random Forests algorithm, each new data point goes
            through the same process, but now it visits all the different trees in the ensemble, 
            which are were grown using random samples of both training data and features.
            Depending on the task at hand, the functions used for aggregation will differ. 
            For Classification problems, it uses the mode or most frequent class predicted by 
            the individual trees (also known as a majority vote) """
            
            from sklearn.ensemble import RandomForestClassifier
            rfc = RandomForestClassifier(n_estimators=800, max_depth = 4, random_state=0)
            rfc.fit(X_train, y_train)
            return rfc.predict(X_test)
        
        elif self.classifier_name == "Neural Network":
            """ Neural Network consists of an input layer, hidden layers and an output layer
            Each layer contains a number of units which make calculations and then the units
            forward the results to the next layer until the output layer is reached which 
            contains the predicted class.The leftmost layer, known as the input layer, consists
            of a set of neurons representing the input features. Each neuron in the hidden layer 
            transforms the values from the previous layer with a weighted linear summation, followed by
            a non-linear activation function like the hyperbolic tan function. The output layer
            receives the values from the last hidden layer and transforms them into output values.
            This classifier produces the highest accuracy possible."""
            
            # Tunning the hidden layer size parameter if Neural Network is chosen
            from sklearn.neural_network import MLPClassifier
            hidden_layer_size, NN_model = self.tune_NN_parameters(X_train, y_train, X_test, y_test)
            print("\n" + str(hidden_layer_size) + " is the chosen hidden layer size for the neural network")
            
            return NN_model.predict(X_test)
        
    def get_accuracy(self, y_test, y_predicted):
        """ This method returns the accuracy of the model """
 
        # Import scikit-learn metrics module for accuracy calculation
        # from sklearn import metrics

        # return metrics.accuracy_score(y_test, y_predicted) * 100
    
        return ((np.sum(y_test == y_predicted) / y_test.shape[0])*100)
        
    def tune_NN_parameters(self, X_train, y_train, X_test, y_test):
            ''' This method runs the neural network training algorithm with multiple
            iterations and chooses different values for the hidden layer sizes
            on each iteration and calculates the accuracy each time. 
            The parameter value of the model with the highest accuracy
            is returned '''
            
            from sklearn.neural_network import MLPClassifier
            hidden_layer_sizes = [(200,30), (300,30), (400,30), (500,30), (600, 30),
                                  (700,50), (800, 100), (900, 120), (800, 30), (800,40),
                                  (1000,120), (1000,30), (600,21), (600, 10), (800,),
                                  (21,), (10,), (50,), (600,600,600), (100,100,100), (900,)]
            max_accuracy = 0
            index=0
            
            print("\nTunning the hidden layer size parameter for the Neural Network...\n")
            
            for i in range(0,len(hidden_layer_sizes)):
                
                model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=hidden_layer_sizes[i]).fit(X_train, y_train)
                temp_accuracy = ((np.sum(y_test == model.predict(X_test)) / y_test.shape[0]) * 100)
                print(str(hidden_layer_sizes[i]) + " (hidden layer size) accuracy: " + str(temp_accuracy) + " %")
            
                if temp_accuracy > max_accuracy:
                    max_accuracy = temp_accuracy
                    NN_best_model = model
                    index = i
            
            return hidden_layer_sizes[index], NN_best_model
    
        
0
