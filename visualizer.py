import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class Visualizer:
    """ This class contains method which help in visualizing the data given or produced
    so  the inputs and results can be understood clearly """

    def plot_results(self, y_test, y_result, method_identifier):
        """ This method plots the actual data against the predicted data """

        if method_identifier == 2: # Plotting the predicted price vs actual price when applying regression
            plt.scatter(y_test, y_result, c='blue')
            plt.title('Actual price vs. Predicted price')
            plt.xlabel('y_test (Actual Price)')
            plt.ylabel('y_predicted (Predicted Price)')
            plt.show()

        elif method_identifier == 1:
            # -------------Classification-----------------------
            # The actual data
            plt.hist(y_test)
            plt.title('Malginant vs. Benign (Actual Data)')
            plt.xlabel('0 = Benign   1 = Malignant')
            plt.ylabel('Number of patients')
            plt.show()

            # The predicted classification
            plt.hist(y_result)
            plt.title('Malginant vs. Benign (Predicted Classification Results)')
            plt.xlabel('0 = Benign   1 = Malignant')
            plt.ylabel('Number of patients')
            plt.show()

    def plot_clustering_without_legend(self, iris):
        # -------------Clustering-------------------------
        pca = PCA(n_components=2).fit(iris.data)
        pca_2d = pca.transform(iris.data)
        plt.figure('Reference Plot')
        plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c=iris.target)
        kmeans = KMeans(n_clusters=3, random_state=111)
        kmeans.fit(iris.data)
        plt.figure('K-means with 3 clusters')
        plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c=kmeans.labels_)
        plt.show()

    def plot_clustering_with_legend(self, iris):
        # -------------Clustering-------------------------
        pca = PCA(n_components=2).fit(iris.data)
        pca_2d = pca.transform(iris.data)
        plt.title('Reference Plot')
        for i in range(0, pca_2d.shape[0]):
            if iris.target[i] == 0:
                c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
            elif iris.target[i] == 1:
                c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')
            elif iris.target[i] == 2:
                 c3 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='*')
        plt.legend([c1, c2, c3], ['Setosa', 'Versicolor', 'Virginica'])
        plt.show()
        kmeans = KMeans(n_clusters=3, random_state=111)
        kmeans.fit(iris.data)
        for i in range(0, pca_2d.shape[0]):
            if kmeans.labels_[i] == 0:
                c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
            elif kmeans.labels_[i] == 1:
                c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')
            elif kmeans.labels_[i] == 2:
                c3 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='*')
        plt.legend([c1, c2, c3], ['Cluster 0', 'Cluster 1', 'Cluster 2'])
        plt.title('K-means clusters the Iris dataset into 3 clusters')
        plt.show()







