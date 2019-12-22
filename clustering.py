class Clustering:
    """ This class applies the K-Means algorithm on the iris dataset to group similar species together  """
    
    def __init__(self,clustering_name):
        self.clustering_name = clustering_name
        
    def cluster(self,X_train,X_test):
        """ This method trains the model and clusters the X_test dataset """
        
        if self.clustering_name == "K-Means":
            
            from sklearn.cluster import KMeans
            
            # Train the model
            trained_model = KMeans(n_clusters=3, random_state=0).fit(X_train)
            
            # Predict
            return trained_model.predict(X_test) 
        
