import numpy as np
import scipy as sp
import os
import scipy.io.arff as arff

#Task 1
def parser(path):
    """
    function which parses the data from an arff file 
    
    @param path: string containig the path to file    
    
    @return array containing the data    
    @raise FileNotFoundError exception in case if the path does not point to a valid file
    """
    
    return data
	
#Task 3
def entropyOnSubset(data, class_label, indices=None):
    """
    function to calculate the entropy of the given dataset and labels
    
    @param data: array containing the data 
    @param class_label: class label of instances
    @param indices: list of indices of the chosen subset from the given complete dataset
            If set to None the entire dataset is used. Default value: indices = None

    @return: the entropy as a float
    """
    
    return entropy


#Task 4
def informationGain(data, class_label, attribute, indices=None):
    """
    function which returns the information gain using a given dataset, indices, and attributes

    @param data: array containing the data
    @param class_label: class label of instances
    @param attribute: selected attribute
    @param indices: list of indices of the chosen subset from the given complete dataset
            If set to None the entire dataset is used. Default value: indices = None
            
    @return: informationGain as a float.
    """
    
    return infoGain


#Task 5
def attributeSelection(data, attributes, class_label, indices=None):
    """
    function which selects a the best attribute for the current node given an array of possible attributes and the data with their labels
    
    @param data: array containing the data
    @param attributes: list of all attributes from which the optmial one should be selected
    @param class_label: class label of instances
    @param indices: list of indices of the chosen subset from the given complete dataset
            If set to None the entire dataset is used. Default value: indices = None
        
    @return: index of the best attribute
    """
    
    return bestIndex


#Task 2
class Node:
    """
    contain the structure of a decision tree: it has either subnodes associated with the corresponding attribute values or is a leaf node. 
    To set the arributes or leaf value use functions, do not access the parameters directly!
    """
    def __init__(self):
        
        
        
    def trainNode(self, data, attributes, class_label, indices=None):
        """
        ID3 based algorithm to find the optimal attribtue
            
        @param data: array containing the data
        @param attributes: list of all attributes from which the optmial one should be selected
        @param indices: list of indices of the chosen subset from the given complete dataset
            If set to None the entire dataset is used. Default value: indices = None
        """
        
                    
           
            
    
    def print_recursive(self, indents):
        """
        function to print the entire tree
        
        @param indents: Integer value of how many indents there should be when printing this node information
        """
        
            
    
class DecisionTree:
    """
    class which represents the decision tree and holds the reference to root node
    """
    def __init__(self):
        self.root = Node()
              
        
    def trainModel(self, data, attributes, class_label):
        """
        function to train the model using a given dataset
        
        @param data: array containing the data with their labels
        @param attributes: list of all attributes from which the optmial one should be selected
        """
        
        
        
    def trainModelOnSubset(self, data, attributes, class_label, indices=None, startNode = None):
        """
        train a certain part of the tree starting with the given startNode based on a subset of the data indicated by the indices array
        
        @param data: array containing the data with their labels
        @param attributes: list of all attributes from which the optmial one should be selected
        @param class_label: list of class values 
        @param indices: list of indices of the chosen subset from the given complete dataset
            If set to None the entire dataset is used. Default value: indices = None
        @param startNode: the root node of the subtree. By default the start node is the root of the tree. Default value: startNode = None
        """
        
        
                            
    def get_classLabel(self, dataset):
        """
        function which returns the expected class value for the given dataset
        """                    
        
        
            
    def print_tree(self):
        """
        function to print the entire tree
        """
        self.root.print_recursive(0)
        
    def get_prediction(self, data):
        """
        function which predicts the class label for the given data
        
        @param data: list of test data
        """
        
		
		
#Task 6: test on dataset
path = 'weather.nominal.arff'
data_arff = parser(path)
#ToDo: Create train and test data and get resulted tree


