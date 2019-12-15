import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class Preprocessor:
    """ This class cleans the data so it can be used for classification """
    
    def __init__(self, X , y):
        self.X = X
        self.y = y
    
    def split_data(self,test_size):
        """ This method splits into training and testing sets based on the test_size parameter """
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = test_size, random_state = 0)
        
        return X_train, X_test, y_train, y_test