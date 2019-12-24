class Clustering:
    """ This class applies the K-Means algorithm on the iris dataset to group similar species together  """

    def __init__(self, clustering_name):
        self.clustering_name = clustering_name

    def cluster(self, X_train, X_test, number_clusters):
        """ This method trains the model and clusters the X_test dataset """

        if self.clustering_name == "K-Means":
            from sklearn.cluster import KMeans

            # Train the model
            trained_model = KMeans(n_clusters= number_clusters, random_state=0).fit(X_train)

            # Predict
            return trained_model.predict(X_test)
        
    def tune_parameters(self, X_train):
            ''' This method runs the training algorithm with multiple
            iterations and chooses different values for the number of clutsers
            on each iteration and calculates the inertia each time. 
            The parameter value of the model with the lowest inertia is returned 
            '''
            
            from sklearn.cluster import KMeans
            n_clusters = [1, 2, 3, 4]
            min_inertia = 9999
            index=0
            
            for i in range(0,len(n_clusters)):
                
                model = KMeans(n_clusters= n_clusters[i], random_state=0).fit(X_train)
                temp_inertia = model.inertia_
                print(temp_inertia)
            
                if temp_inertia < min_inertia:
                    min_inertia = temp_inertia
                    index = i
            
            return n_clusters[index]
                
                
                
                
                
