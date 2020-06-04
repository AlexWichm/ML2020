import DecisionTree as tree # Decision tree implementaion from last exercise
import numpy as np
import scipy as sp
import os

        
class AdaBoostTree:
    
    def __init__(self, k):
     """
        k: maximum number of iterations (ensemble size)
     """   
    
    
    #Task 1
    def sampling(self, data, weights, length=-1):
    """
    function to sample from the instance set based on the given weights

    @param data: array containing the data
    @param weights: array of weights of the instances
    @param length: the length of the returned array. If set to -1 the
            length of the given data array will be used. Default: length=-1
            
    @return: array containing the given data with the occurrences of the samples based on their weights
    """
        
        return new_data
    
    
    #Task 2
    def modelGeneration(self, data, attributes, class_label):
        """
        function to generate the trees sequentially, each new tree gets the weighted dataset which was calculated based on the errors of the previous tree.
        All wrongly classified class labels get an increased weight, leading to a higher occurrences of their samples.

        @param data: array containing the data
        @param attributes: list of all possible attributes
        @param class_label: class label of instances
        """
        
        
     
    #Task 3
    def classification(self, sample):
        """
        function to classify a given sample using the trained model
        
        @param sample: the test data
        
        @return: the predicted class label
        """
        
        
        return predicted_class
    
    



# Task 4--7
path = 'car.arff'

