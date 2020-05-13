import numpy as np
import scipy as sp
import os
import scipy.io.arff as arff

import pandas 
import random

#Task 1

class Arff:
	
	relation = ""
	data = []
	attribute = {}
	
	def __init__(self, relation = None, attribute = None, data = None):
		self.relation = relation if (relation is not None) else ""
		self.attribute = attribute if (attribute is not None) else {}
		self.data = data if (data is not None) else pandas.DataFrame(np.array([]))
	
	def read_arff(self, path):
		start = 0
		in1 = open(path, "r")

		dat = []
		for x in in1:
			if x.startswith('@relation'):
				self.relation = x.split(" ")[1].replace("\n","")
			if x.startswith('@attribute'):
				tmp = x.replace("{","").replace("}","").replace("\n","")
				#checks if whitespaces between commas in attributes occur
				if ( len(tmp.split(" ")) > 3):
					values = tmp.replace(",","").split(" ")[2:]
				else:
					values = tmp.split(" ")[2].split(",")
				
				self.attribute.update({tmp.split(" ")[1]:values})
			if start:
				dat.append(x.replace("\n","").split(","))
			if x.startswith('@data'):
				start=1
		
		self.data = pandas.DataFrame(np.array(dat), columns = self.attribute, index = range(0, len(dat)) )
	
	
	def print(self):
		if self.relation == "":
			print("Dataset is empty")
		else:
			print("Relation: " + self.relation)
			print("Attribute: " + str(self.attribute))
			print("Data: " + str(self.data))
	
	def clone(self):
		return Arff(self.relation, self.attribute, self.data)

		
def parser(path):
	"""
	function which parses the data from an arff file 
	
	@param path: string containig the path to file	
	
	@return array containing the data	
	@raise FileNotFoundError exception in case if the path does not point to a valid file
	"""
	
	data = Arff()
	data.read_arff(path)
	
	return data
	
	
def splitData(data, class_label, seed, ratio):
	"""
	function to split a dataset into train and test parts using a provided initial random seed. 
	
	@param data: array containing the data 
	@param class_label: class label of instances
	@param seed: an input random seed 
	@param ratio: a float number indicating the ratio of training data  
	
	@return split_list containing the list of training and test data and their labels
	"""
	
	random.seed(seed)
	subset = data.clone()
	size_data = subset.data.shape[0]
	n = int(np.floor(size_data * ratio)) # number of datasets in train
	index =  random.sample(range(1, size_data), n)
	split_list = [item for item in [0] for i in range(size_data)]
	
	for i in index:
		split_list[i]=1
	
	return split_list #returns list of indeces where 0 is test and 1 is training data 

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
		pV = subset.count(i)/len(subset)
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
	subset = data[:] if indices == None else data.loc[indices]
	
	sublist = subset[attribute].tolist()
	values = list(set(sublist))
	infoGain = entropyOnSubset(subset, class_label)
	
	#print (sublist)
	
	for i in values:
		index = list(subset.index[subset[attribute] == i])
		infoGain -= sublist.count(i)/len(sublist) * entropyOnSubset(subset, class_label, index)

	
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
			i.print_recursive(indents+1)
			
	def get_bebe(self, label):
		#print("Name: " + self.name)
		#print(label)
		#print(len(self.children))
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
	def __init__(self):
		self.root = Node()
			  
		
	def trainModel(self, data, attributes, class_label, max_depth=-1):
		"""
		function to train the model using a given dataset
		
		@param data: array containing the data with their labels
		@param attributes: list of all attributes from which the optmial one should be selected
		"""
		
		self.trainModelOnSubset(data, attributes, class_label, None, None,  max_depth)
		
		
		
	def trainModelOnSubset(self, data, attributes, class_label, indices=None, startNode = None, max_depth=-1):
		"""
		train a certain part of the tree starting with the given startNode based on a subset of the data indicated by the indices array
		
		@param data: array containing the data with their labels
		@param attributes: list of all attributes from which the optmial one should be selected
		@param class_label: list of class values 
		@param indices: list of indices of the chosen subset from the given complete dataset
			If set to None the entire dataset is used. Default value: indices = None
		@param startNode: the root node of the subtree. By default the start node is the root of the tree. Default value: startNode = None
		"""
		
		#print (data)
		#print("_________---_________")
		
		if startNode == None:
			startNode = self.root
		
		#print(startNode.name)
		
		#print("Attributes: " + str(attributes))
		
		new_attributes=attributes[:]
		
		if indices == None:
			subset = data			
		else:
			subset = data.loc[indices]
		
		
		if (len(new_attributes) > 0 and not subset.empty):
			bestAttribute_index = startNode.trainNode(data, new_attributes, class_label, indices)
			bestAttribute = new_attributes[bestAttribute_index]
			new_attributes.pop(bestAttribute_index)
			values = attributes_full[bestAttribute]
			
			
			
			
			for i in values:
				node = Node(i)
				startNode.add_child(node)
				#print(node.get_name())
				#print(bestAttribute)
				index = list(subset.index[subset[bestAttribute] == i])
				self.trainModelOnSubset(subset, new_attributes, class_label, index, node)
			
		else:
			bestAttribute=class_label
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
		accuracy = 0
		hit=0
		count=0
		for index, row in test.iterrows():
			count += 1
			tmp = self.get_classLabel(row.tolist(),row[class_label])
			#print (tmp)
			if tmp:
				hit+=1
		#print ("hit "+ str(hit) )
		accuracy = hit/count
		
		return accuracy
	
	def get_classLabel(self, dataset, class_label):
		"""
		function which returns the expected class value for the given dataset
		"""     
		node = self.root
		broken=0
		
		#print("BEBE:" + str(node.get_bebe( dataset)))
		
		if (node.get_bebe( dataset) == class_label ):
			return 1
		else:
			return 0

			def junk(data, class_label, seed, ratio):	
	random.seed(seed)
	size_data = len(data)
	train = []
	test = []
	n = int(np.floor(size_data * ratio)) # number of datasets in train
	index =  random.sample(range(1, size_data), n)
	split = [item for item in [0] for i in range(size_data)]

	for i in index:
		split[i]=1
	
	count = 0
	for i in range(0,len(data)): #splitting data in test and train data
		if split[i]:
			train.append(data[i])
			count+=1
		else:
			test.append(data[i])
	split_list = [ test, train ]
	return split_list #returns list of indeces where 0 is test and 1 is training data 



#Task 2: test on car dataset in a loop
path = 'car.arff'
data_arff = parser(path)

data_arff.print()
print(data_arff.attribute)

print(splitData(data_arff, "class", 23, 0.5))
print("Information Gain on maint: " + str(informationGain(data_arff, "class", "maint")))


#Task 3 & 5

