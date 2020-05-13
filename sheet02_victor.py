import numpy as np
import scipy as sp
import os
import scipy.io.arff as arff

import random
import sys
import pandas

# global_variable
attributes_full = {}

#Task 1
def splitData(data, class_label, seed, ratio):

    """
    function to split a dataset into train and test parts using a provided initial random seed. 
    
    @param data: array containing the data 
    @param class_label: class label of instances  
    @param seed: an input random seed 
    @param ratio: a float number indicating the ratio of training data  
    
    @return split_list containing the list of training and test data and their labels
    """
    split_list = []
    split_list.append(data.sample(frac=ratio,random_state=seed))
    split_list.append(data.drop(split_list[0].index))

    return split_list



def parser(path):
    """
    function which parses the data from an arff file

    @param path: string containig the path to file

    @return array containing the data
    @raise FileNotFoundError exception in case if the path does not point to a valid file
    """

    start = 0
    data = []
    attributes = []
    counter = 0
    if os.path.isfile(path):
        in1 = open(path, "r")
        for x in in1:
            if x.startswith('@attribute'):
                tmp = x.replace("{", "").replace("}", "").replace("\n", "").replace(",", " ").split(" ")    #had to change the parser replacing , with "" for this particular dataset
                values = tmp[2:]
                attributes.append(tmp[1])
                attributes_full.update({tmp[1]: values})
            if start:
                data.append(x.replace("\n", "").split(","))
            if x.startswith('@data'):
                start = 1
        return pandas.DataFrame(np.array(data), columns=attributes, index=range(0, len(data)))
    else:
        sys.exit("FileNotFoundError")


def entropyOnSubset(data, class_label, indices=None):
    """
    function to calculate the entropy of the given dataset and labels

    @param data: array containing the data
    @param class_label: class label of instances
    @param indices: list of indices of the chosen subset from the given complete dataset
            If set to None the entire dataset is used. Default value: indices = None
    @return: the entropy as a float
    """
    subset = []
    if indices == None:
        subset = data[:]
    else:
        subset = data.loc[indices]

    subset = subset[class_label].tolist()
    values = list(set(subset))
    entropy = 0

    for i in values:
        pV = subset.count(i) / len(subset)
        entropy -= pV * np.log2(pV)

    return entropy


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
    subset = []
    if indices == None:
        subset = data[:]
    else:
        subset = data.loc[indices]

    sublist = subset[attribute].tolist()
    values = list(set(sublist))
    infoGain = entropyOnSubset(subset, class_label)

    # print (sublist)

    for i in values:
        index = list(subset.index[subset[attribute] == i])
        infoGain -= sublist.count(i) / len(sublist) * entropyOnSubset(subset, class_label, index)

    return infoGain


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
    best = 0
    bestIndex = 0
    counter = 0
    for i in attributes:
        infoG = informationGain(data, class_label, i, indices)
        if infoG > best:
            best = infoG
            bestIndex = counter
        counter += 1

    return bestIndex


class Node:
    """
    contain the structure of a decision tree: it has either subnodes associated with the corresponding attribute values or is a leaf node.
    To set the arributes or leaf value use functions, do not access the parameters directly!
    """

    def __init__(self, name="root", children=None):
        self.name = name
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)

    def add_child(self, child):
        self.children.append(child)

    def get_name(self):
        return self.name()

    def trainNode(self, data, attributes, class_label, indices=None):
        """
        ID3 based algorithm to find the optimal attribtue

        @param data: array containing the data
        @param attributes: list of all attributes from which the optmial one should be selected
        @param indices: list of indices of the chosen subset from the given complete dataset
            If set to None the entire dataset is used. Default value: indices = None
        """
        optimalAttribute = attributeSelection(data, attributes, class_label, indices)

        return optimalAttribute

    def print_recursive(self, indents):
        """
        function to print the entire tree

        @param indents: Integer value of how many indents there should be when printing this node information
        """

        ind = "\t"
        output = indents * ind + self.name
        print(output)
        for i in self.children:
            i.print_recursive(indents + 1)

    def get_bebe(self, label):
        # print("Name: " + self.name)
        # print(label)
        # print(len(self.children))
        clone = label[:]
        if len(self.children) == 0:
            return str(self.name)
        for j in clone:
            for i in self.children:
                if i.name == j:
                    clone.remove(j)
                    return i.get_bebe(clone)




class DecisionTree:
    """
    class which represents the decision tree and holds the reference to root node
    """

    def __init__(self, max_depth = -1):
        self.root = Node()
        self.max_depth = max_depth


    def trainModel(self, data, attributes, class_label):
        """
        function to train the model using a given dataset

        @param data: array containing the data with their labels
        @param attributes: list of all attributes from which the optmial one should be selected
        """

        self.trainModelOnSubset(data, attributes, class_label)


    def trainModelOnSubset(self, data, attributes, class_label, indices=None, startNode=None, depth = 0):
        """
        train a certain part of the tree starting with the given startNode based on a subset of the data indicated by the indices array

        @param data: array containing the data with their labels
        @param attributes: list of all attributes from which the optmial one should be selected
        @param class_label: list of class values
        @param indices: list of indices of the chosen subset from the given complete dataset
            If set to None the entire dataset is used. Default value: indices = None
        @param startNode: the root node of the subtree. By default the start node is the root of the tree. Default value: startNode = None
        """

        # print (data)
        #print("_________---_________")

        if startNode == None:
            startNode = self.root

        #print(startNode.name)

        #print("Attributes: " + str(attributes))

        new_attributes = attributes[:]

        if indices == None:
            subset = data
        else:
            subset = data.loc[indices]

        if ((len(new_attributes) > 0 and not subset.empty) and (depth < self.max_depth or self.max_depth == -1)):
            bestAttribute_index = startNode.trainNode(data, new_attributes, class_label, indices)
            bestAttribute = new_attributes[bestAttribute_index]
            new_attributes.pop(bestAttribute_index)
            values = attributes_full[bestAttribute]

            for i in values:
                node = Node(i)
                startNode.add_child(node)
                #print(node.name)
                #print(bestAttribute)
                index = list(subset.index[subset[bestAttribute] == i])
                self.trainModelOnSubset(subset, new_attributes, class_label, index, node,depth+1)

        else:
            bestAttribute = class_label
            values = attributes_full[bestAttribute]
            sublist = data[bestAttribute].tolist()
            top = 0
            name = "test"
            for i in values:
                if sublist.count(i) > top:
                    name = i
                    top = sublist.count(i)
            node = Node(name)
            startNode.add_child(node)

    """
    function which returns the expected class value for the given dataset
    """

    def print_tree(self):
        """
        function to print the entire tree
        """
        self.root.print_recursive(0)

    def get_prediction(self, data, class_label):
        """
        function which predicts the class label for the given data

        @param data: list of test data
        """
        test = data.loc[:]
        accuracy = 0
        hit = 0
        count = 0
        for index, row in test.iterrows():
            count += 1
            tmp = self.get_classLabel(row.tolist(), row[class_label])
            # print (tmp)
            if tmp:
                hit += 1
        # print ("hit "+ str(hit) )
        accuracy = hit / count

        return accuracy

    def get_classLabel(self, dataset, class_label):
        """
        function which returns the expected class value for the given dataset
        """
        node = self.root
        broken = 0

        # print("BEBE:" + str(node.get_bebe( dataset)))

        if (node.get_bebe(dataset) == class_label):
            return 1
        else:
            return 0





path = 'car.arff'
data_arff = parser(path)
c_label = "class"
attributes = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]

#Task1
train_data, test_data = splitData(data_arff,c_label,123,0.7)
tree = DecisionTree()
tree.trainModel(train_data, attributes, c_label)
#print("Accuracy: " + str(tree.get_prediction(test_data,c_label)))


#Task 2: test on car dataset in a loop
def acc_loop(n,ratio):    #takes pretty long
    results = []
    for i in range(n):
        train,test = splitData(data_arff,c_label,i,ratio)
        t = DecisionTree()
        t.trainModel(train,attributes,c_label)
        results.append(t.get_prediction(test,c_label))
    return results
def printResults(n,ratio):
    accuracies = acc_loop(n,ratio)
    print("n: "+str(n)+ " ratio: "+ str(ratio))
    print("Mean Accuracy: "+str(np.mean(accuracies)))
    print("Standard deviation: "+str(np.std(a = accuracies))+"\n")
#printResults(10,0.2)

#Task 3 & 5

#takes VERY long
#printResults(10,0.5)
#printResults(10,2/3)
#printResults(10,0.75)
#printResults(10,0.9)
#the change in ratio seems to make no significant difference


def acc_loop5(depth,n,ratio):    #takes pretty long
    results = []
    for i in range(n):
        train,test = splitData(data_arff,c_label,i,ratio)
        t = DecisionTree(depth)
        t.trainModel(train,attributes,c_label)
        results.append(t.get_prediction(test,c_label))
    return results
def printResults5(depth,n,ratio):
    accuracies = acc_loop5(depth,n,ratio)
    print("depth: "+str(depth)+"n: "+str(n)+ " ratio: "+ str(ratio))
    print("Mean Accuracy: "+str(np.mean(accuracies)))
    print("Standard deviation: "+str(np.std(a = accuracies))+"\n")

printResults5(1,10,0.5)
printResults5(1,10,2/3)
printResults5(1,10,0.75)
printResults5(1,10,0.9)

printResults5(3,10,0.5)
printResults5(3,10,2/3)
printResults5(3,10,0.75)
printResults5(3,10,0.9)

printResults5(5,10,0.5)
printResults5(5,10,2/3)
printResults5(5,10,0.75)
printResults5(5,10,0.9)




