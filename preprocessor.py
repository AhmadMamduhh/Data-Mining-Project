import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Preprocessor:
    """ This class cleans the data so it can be used for classification, regression or clustering """
    
        
    
    def split_data(self, X, test_ratio, y=None):
        """ This method splits into training and testing sets based on the test_size parameter """
        
        if type(y) == type(None):
            X_train, X_test = train_test_split(X, test_size = test_ratio, random_state = 0)
            return X_train, X_test
        
        else:    
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_ratio, random_state = 0)
            return X_train, X_test, np.ravel(y_train), np.ravel(y_test)
            
        
    
    def scaling(self, X_train, X_test, scale_type):
        """This method applies standard scaler, min max scaler, max scaler or min scaler based on
        a parameter (scale_type) using sklearn library""" 
        
        if scale_type == 'Standard Scaler':
            from sklearn.preprocessing import StandardScaler
            standard_scaler = StandardScaler()
            standard_scaler.fit(X_train)
            X_train = standard_scaler.transform(X_train)
            X_test = standard_scaler.transform(X_test)
            return X_train, X_test
        
        elif scale_type == 'Min Max Scaler':
            from sklearn.preprocessing import MinMaxScaler
            min_max_scaler = MinMaxScaler()
            min_max_scaler.fit(X_train)
            X_train = min_max_scaler.transform(X_train)
            X_test = min_max_scaler.transform(X_test)
            return X_train, X_test
            
        elif scale_type == 'Max Scaler':
            from sklearn.preprocessing import MaxAbsScaler
            max_scaler = MaxAbsScaler()
            max_scaler.fit(X_train)
            X_train = max_scaler.transform(X_train)
            X_test = max_scaler.transform(X_test)
            return X_train, X_test
        
        elif scale_type == 'Robust Scaler':
            # This scaler uses percentiles to prevent the outlier values from influencing the scaling
            from sklearn.preprocessing import RobustScaler
            robust_scaler = RobustScaler(quantile_range=(25, 75))
            robust_scaler.fit(X_train)
            X_train = robust_scaler.transform(X_train)
            X_test = robust_scaler.transform(X_test)
            return X_train, X_test

        elif scale_type == 'Min Scaler':
            min_value = np.min(X_train, axis=0).reshape(X_train.shape[1], 1).T
            X_train = min_value / (X_train + 0.000001) # to avoid division by zero errors
            X_test = min_value / (X_test + 0.000001) # to avoid division by zero errors
            
            return X_train.astype(np.float64), X_test.astype(np.float64)
                
            
      
    def encoding(self, X, y, dataset_name):
        """ This method encodes the categorical data """
        from sklearn.preprocessing import LabelEncoder
        
        # Encoding the categorical data in the diamonds dataset
        if dataset_name == 'diamonds':
            labelencoder = LabelEncoder()
            X[:,1] = labelencoder.fit_transform(X[:,1])
            X[:,2] = labelencoder.fit_transform(X[:,2])
            X[:,3] = labelencoder.fit_transform(X[:,3])
            
        # Encoding the categorical data in the breast cancer dataset
        elif dataset_name == 'breast cancer':
            labelencoder = LabelEncoder()
            y[:,0] = labelencoder.fit_transform(y[:,0])
            y=y.astype(np.int64)
        
        return X, y
            

    
    def drop_missing(self, dataset):
        """ This method drops missing entries (rows) using pandas"""
        if type(dataset) == type(pd.DataFrame()):
            dataset.dropna(inplace=True, axis=0)
            
        elif type(dataset) == type(np.array):
            dataset = pd.DataFrame(dataset)
            dataset.dropna(inplace=True, axis=0)
            dataset = dataset.to_numpy()
            
        return dataset
        
    def replace_missing(self, dataset):
        """ This method replaces missing entries with the mode of the column """
        
        if type(dataset) == type(pd.DataFrame()):
            dataset.fillna(dataset.mode(), inplace=True, axis=0)
            
        elif type(dataset) == type(np.array):
            dataset = pd.DataFrame(dataset)
            dataset.fillna(dataset.mode(), inplace=True, axis=0)
            dataset = dataset.to_numpy()
            
        return dataset
        
    def dataframe_to_numpy(self, dataset, dataset_name):
        """ This method takes the dataset and splits it into two numpy arrays.
            One for the features and one for the output labels. """
            
        if dataset_name == 'diamonds':
            dataset.drop(dataset.columns[0], axis='columns', inplace = True)
            X = dataset.drop(labels='price', axis=1).iloc[:, 0:8].to_numpy()
            y = dataset.iloc[:, 6].values.reshape(dataset.shape[0],1)
            y=y.astype(np.float64)
            
        elif dataset_name == 'breast cancer':
            # Dropping the id column
            dataset.drop(dataset.columns[0], axis='columns', inplace = True)
            
            X = dataset.iloc[:,1::].to_numpy()
            y = dataset.iloc[:, 0].values.reshape(dataset.shape[0],1)
        
        else:
           
            return dataset
        
        return X,y