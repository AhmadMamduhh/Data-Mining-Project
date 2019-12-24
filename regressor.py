

class Regressor:
    """ This class contains linear regression, polynomial regression, decision tree, KNN regressor
    and Neural Network MLP Regression """
    
    def __init__(self, regressor_name):
      self.regressor_name = regressor_name
    

    def predict(self, X_train, y_train, X_test):
        """ This method uses the apropriate regression algorithm based on
        the value of the regression_name string and uses the X_test 
        parameter and to predict y_predicted. X_train and y_train are the data that
        are used for fitting. """
        
        
        if self.regressor_name == 'KNN Regression':
            from sklearn.neighbors import KNeighborsRegressor
            neigh = KNeighborsRegressor(n_neighbors=6)
            neigh.fit(X_train, y_train)
            return neigh.predict(X_test)

            
        elif self.regressor_name == 'Decision Tree':
            from sklearn.tree import DecisionTreeRegressor
            decision_tree = DecisionTreeRegressor(random_state=0, max_depth=3)
            decision_tree.fit(X_train, y_train)
            return decision_tree.predict(X_test)
        
        elif self.regressor_name == 'Linear Regression':
            from sklearn.linear_model import LinearRegression
            linear_regressor = LinearRegression()
            linear_regressor.fit(X_train, y_train)
            return linear_regressor.predict(X_test)
        
        elif self.regressor_name == 'Polynomial Regression':
            from sklearn.preprocessing import PolynomialFeatures  
            from sklearn.linear_model import LinearRegression
            
            poly_features = PolynomialFeatures(degree = 2)  
            X_poly = poly_features.fit_transform(X_train)
            X_poly_test = poly_features.fit_transform(X_test)
            poly_model = LinearRegression()  
            poly_model.fit(X_poly, y_train)
            
            return poly_model.predict(X_poly_test)
            
  
            
        
        elif self.regressor_name == "Random Forest":
            """ Random Forest regressor is based on the decision tree regressor.
            The random forest is an algorithm consisting of many decisions trees.
            It uses bagging and feature randomness when building each individual tree to try to create 
            an uncorrelated forest of trees whose prediction by committee is more accurate than 
            that of any individual tree. In the Random Forests algorithm, each new data point goes
            through the same process, but now it visits all the different trees in the ensemble, 
            which are were grown using random samples of both training data and features.
            Depending on the task at hand, the functions used for aggregation will differ. 
            For regression problems, it uses the mean of all the tree results to get the output
            value """
            
            from sklearn.ensemble import RandomForestRegressor
            rfc = RandomForestRegressor(n_estimators=300, max_depth = 4, random_state=0)
            rfc.fit(X_train, y_train)
            return rfc.predict(X_test)
        
        elif self.regressor_name == "Neural Network":
            """ Neural Network consists of an input layer, hidden layers and an output layer
            Each layer contains a number of units which make calculations and then the units
            forward the results to the next layer until the output layer is reached which 
            contains the predicted class.The leftmost layer, known as the input layer, consists
            of a set of neurons representing the input features. Each neuron in the hidden layer 
            transforms the values from the previous layer with a weighted linear summation, followed by
            a non-linear activation function like the hyperbolic tan function. The output layer
            receives the values from the last hidden layer and transforms them into output values.
            This regressor produces the highest accuracy possible."""
            
            from sklearn.neural_network import MLPRegressor
            NN = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (200, X_test.shape[1]), random_state=0)
            NN.fit(X_train, y_train)
            return NN.predict(X_test)
        
    def get_accuracy(self, y_test, y_predicted):
        """ This method returns the accuracy of the model """
 
        # Import scikit-learn metrics module for accuracy calculation
        from sklearn import metrics

        return metrics.r2_score(y_test, y_predicted)
    
        
        
        