import numpy as np
import pandas as pd

class Classifier:
    """ An instance of this class is given a certain classifier name when
        initialized. Based on this name, a certain classification algorithm 
        will be used when the classify method is called. The class takes the
        training data which would be used to fit the classification algorithms
    """
        
    def __init__(self, classifier_name, X , y):
      self.classifier_name = classifier_name
      self.X = X
      self.y = y
    

    def classify(self, X):
        """ This method uses the apropriate classification algorithm based on
        the value of the classifier_name string and classifies the X parameter and returns y_predicted  """
        
        
        if self.classifier_name == 'KNN':
            from sklearn.neighbors import KNeighborsClassifier
            neigh = KNeighborsClassifier(n_neighbors=3)
            neigh.fit(self.X, self.y)
            return neigh.predict(X)

            
        elif self.classifier_name == 'Decision Tree':
            from sklearn.tree import DecisionTreeClassifier
            decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
            decision_tree.fit(self.X, self.y)
            return decision_tree.predict(X)
        
        
