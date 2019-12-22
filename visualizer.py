import matplotlib.pyplot as plt

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
            
     '''   elif method_identifier == 3:
            # -------------Clustering-------------------------
            print('code')
            
     '''