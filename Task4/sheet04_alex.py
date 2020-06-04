import numpy as np
import scipy as sp
import os

import scipy.io.arff as arff
from id3 import Id3Estimator,export_text	 #in id3/export change  from "sklearn.externals import six"  to "import six"
import random
import pandas
import sys
import matplotlib.pyplot as plt


#in order to remove id3 future warnings (BAD style)
import warnings
warnings.filterwarnings("ignore")

def parser(path):
	"""
	function which parses the data from an arff file
	@param path: string containig the path to file

	@return array containing the data
	@raise FileNotFoundError exception in case if the path does not point to a valid file
	"""

	start = 0  # check if data really occured
	# Declaratives as constant to avoid misspelling in code
	RELATION = 'relation'
	ATTRIBUTE = 'attribute'
	DATA = 'data'

	# Create dictionary holding the arff information
	data = {RELATION: [],
			ATTRIBUTE: [],
			DATA: []}

	# Read the file and analyse the data
	with open(path) as file:
		for line in file.readlines():
			# Check if line is empty
			if line.strip() == '':
				continue

			# Check if line contains the relation
			elif '@' + RELATION in line:
				data[RELATION].append(line.replace('@' + RELATION, '').strip())

			# Check if line contains an attribute
			elif line.startswith('@attribute'):
				tmp = line.replace("{", "").replace("}", "").replace("\n", "").replace("'", "")
				# checks if whitespaces between commas in attributes occur
				if (len(tmp.split(" ")) > 3):
					values = tmp.replace(",", "").split(" ")[2:]
				else:
					values = tmp.split(" ")[2].split(",")

				data[ATTRIBUTE].append({'name': tmp.split(" ")[1], 'values': values})

			# check if @data exists
			elif '@' + DATA in line:
				start = 1

			# If the line is not one of the others, it has to be data
			elif '@' + DATA not in line and start:
				line = line.split(',')
				# strip each element of the line
				for i in range(len(line)):
					line[i] = line[i].strip()
				# Add data to dictionary
				data[DATA].append(line)

	attributes = np.array(data['attribute'])
	out = []
	for i in range(len(data['data'])):
		data_dict = {}
		for j in range(len(attributes)):
			data_dict.update({attributes[j]['name']:data['data'][i][j]})
		out.append(data_dict)
	out = np.array(out)
	return out, data[ATTRIBUTE]

def stratification(data):
	'''
	split np array of dictionaries like
	{'buying': 'low', 'maint': 'low', 'doors': '5more', 'persons': 'more', 'lug_boot': 'big', 'safety': 'low', 'class': 'unacc'}
	into datasets of one class attribute each
	'''
	class_attributes = []
	class_sets = []
	for i in range(len(data)):
		contained = False
		for j in range(len(class_attributes)):
			if class_attributes[j]==data[i]['class']:
				class_sets[j].append(data[i])
				contained = True
		if not contained:
			class_attributes.append(data[i]['class'])
			class_sets.append([])

	for i in range(len(class_sets)):
		class_sets[i] = np.array(class_sets[i])
	return class_sets


def makeArff(rel_name, attributes,data):
	filename = rel_name + ".arff"
	arff_file = open(filename,"w+")

	#write rel name
	arff_file.write("@relation "+str(rel_name)+"\n\n")
	#write attributes

	for i in range(len(attributes)):
		#get individual att values
		v = []
		for j in range(len(attributes[i]['values'])):
			v.append(attributes[i]['values'][j])
		arff_file.write("@attribute " + attributes[i]['name']+ " {" +str(v)[1:-1].replace("'","").replace(" ","")+ "}\n")
	arff_file.write("\n@data\n")

	#write data

	for i in range(len(data)):
		d_str = ""
		for j in range(len(attributes)):
			d_str += str(data[i][attributes[j]['name']]) + ","
		arff_file.write(d_str[0:-1]+"\n")

# mildly infuriating splitting those in separate methods; is there a special reason?		
def trainCV(data,attributes,fold):
	makeArff("train"+str(fold),attributes,data)

def testCV(data,attributes,fold):
	makeArff("test"+str(fold),attributes,data)


def stratifiedCrossValidation2(data,attributes,folds):

	np.random.shuffle(data)
	strat_data = stratification(data)
	ratios = []
	pos = []
	data_len = len(data)
	
	for i in (strat_data):
		if (len(i)//folds > 0):
			ratios.append(len(i)//folds) #get number of instances per fold per set
		else:
			ratios.append(1)
		pos.append(0)
	
	tmp_test = []
	tmp_train = []
	
	for i in range(folds):
		
		for j in range(len(strat_data)): #subsetting a portion for every fold of input data
			new_pos = pos[j] + ratios[j]
			if new_pos < len(strat_data[j]):
				
				if i == folds: #to get the 'forgotten' rounded data
					tmp_test.extend(strat_data[j][pos[j]:len(strat_data[j])])
				else:
					tmp_test.extend(strat_data[j][pos[j]:new_pos])
				
				tmp_train.extend(strat_data[j][:pos[j]])
				tmp_train.extend(strat_data[j][new_pos:])
				pos[j] = new_pos
			else:
				tmp_test.extend(strat_data[j][pos[j]:len(strat_data[j])])
				tmp_train.extend(strat_data[j][:pos[j]])
		
		trainCV(tmp_train,attributes,i)
		testCV(tmp_test,attributes,i)
		
		tmp_test.clear()
		tmp_train.clear()
		


def stratifiedCrossValidation(strat_data,attributes,folds):
	distro = []
	sets = []
	data = []
	for i in range(len(strat_data)):
		data.append([])
		data[i] = strat_data[i].tolist()

	for i in range(len(data)):
		#one third test data
		#scale distro to folds*1/3 the size and round
		distro.append(len(data[i])//(folds*3))
		if distro[i] == 0: distro[i]=1  #to avoid rare classes being ignored
	#pick the sets
	for i in range(folds):
		sets.append([])

		for j in range(len(data)):
			for k in range(distro[j]):
				#pick elements for distro
				item = random.choice(data[j])
				sets[i].append(item) #"take" item
				data[j].remove(item)

	for i in range(folds):
		makeArff("test"+str(i),attributes,sets[i])

	train = []
	for i in range(len(data)):		#concatenate stratified arrays back for training set
		for j in range(len(data[i])):
			train.append(data[i][j])
	makeArff("train",attributes,train)
	print(distro)

def splitData(raw,atts):
	data = []
	target = []
	for k in range(len(raw)):
		data.append([])
		for i in range(len(atts)):
			data[k].append(raw[k][atts[i]['name']])
		target.append(data[k][len(atts)-1])
		data[k] = np.array(data[k][:len(atts)-1])
	target = np.array(target)
	data = np.array(data)
	#print(data)
	#print(target)

	return data,target

def writePredictions(predictions,fold):
	pred_file = open("predictions"+str(fold)+".txt","w+")
	for i in range(len(predictions)):
		pred_file.write(str(predictions[i])+"\n")

def CV_learn(classifier,trainPaths,testPaths,folds,atts):
	accs = []
	for i in range(folds):
		tr_raw, tr_att = parser(trainPaths[i])
		te_raw, te_att = parser(testPaths[i])
		#split tr_data in data and target for id3
		tr_data, tr_target = splitData(tr_raw,tr_att)
		te_data, te_target = splitData(te_raw,te_att)
	
		classifier.fit(tr_data,tr_target)

		#print(export_text(classifier.tree_, ['buying ','maint ','doors ','persons ','lug_boot ','safety ']))

		predictions = classifier.predict(te_data)
		writePredictions(predictions,i)
		ctr = 0
		for i in range(len(predictions)):
			if predictions[i] != te_target[i]:
				ctr+=1
		accs.append(  (len(predictions)-ctr)/len(predictions))
	print("mean accuracy: " + str(np.mean(np.array(accs))))
	print("standard deviation: " + str(np.std(np.array(accs))))
	return np.mean(np.array(accs)), np.std(np.array(accs))


def CVparameterSelection(data, option, p_range):
	'''

	 return accuracy of option /safety lug_boot etc ?
	 test for each value in range
	'''
	
	means = []
	stds = []
	max = 0
	
	data, att_data = parser("car.arff")
	stratifiedCrossValidation2(data,att_data,10)
	
	
	test_paths = []#prepare test paths
	train_paths = []#prepare train paths
	
	for i in range(10):
		test_paths.append("test"+str(i)+".arff")
		train_paths.append("train"+str(i)+".arff")
	
	if p_range.startswith("{"): #nominal
		params = p_range.replace("{","").replace("}","").split(",")
		
		
		for counter in params:
			
			if option == "prune":
				estimator = Id3Estimator(prune=counter)
			elif option == "gain_ratio":
				estimator = Id3Estimator(gain_ratio=counter)
			elif option == "is_repeating":
				estimator = Id3Estimator(is_repeating=counter)
			else:
				print("wrong params")
				return None
			
			
			print(str(option)+"="+str(counter))
			CV_learn(estimator,train_paths,test_paths,10,att_data) 
			means.append(tmp[0])
			stds.append(tmp[1])
			counter +=params[1]
			
			if tmp[0] > max:
				max = tmp[0]
				best = counter
				
	else: #real
		params = p_range.split(":")
		counter = int(params[0])
		#print("params: "+ str(params))
		while (counter <= int(params[2])):
			#print(counter)
			
			
			if option == "max_depth":
				estimator = Id3Estimator(max_depth=counter)
			elif option == "min_samples_split":
				estimator = Id3Estimator(min_samples_split=counter)
			elif option == "min_entropy_decrease	":
				estimator = Id3Estimator(min_entropy_decrease=counter)
			else:
				print("wrong params")
				return None
			
			print(str(option)+"="+str(counter))
			tmp = CV_learn(estimator,train_paths,test_paths,10,att_data) #TASK 1 a-e
			
			means.append(tmp[0])
			stds.append(tmp[1])
			
			if tmp[0] > max:
				max = tmp[0]
				best = counter
			
			counter += int(params[1])
			
	return means, stds, best


def OptimalID3(data_path, p_range):
	
	data, att_data = parser(data_path)
	option = "max_depth" 
	
	return CVparameterSelection(data, option, p_range) #Task1 f)
	


	
def Task1():
	
	p_range = "1:1:10"
	means,stds,best = OptimalID3("car.arff", p_range)  #Task1 g)
	
	print("Best : " +str(best))

	plt.plot(means)  #Task1 h)
	plt.ylabel('accuracy')
	plt.xlabel('max_depth')
	plt.show()
	
	'''
	Die Accuracy nimmt mit zunehmender Tiefe zu. Bei max_depth von 5 hat der Baum die maximale Accuracy erreicht und sie bleibt ab 5 gleich. 
	
	'''
	
def Task3():
	dia, dia_meta = parser("diabetes.arff")
	estimator = Id3Estimator()

	stratifiedCrossValidation2(dia, dia_meta,10)
	estimator = Id3Estimator()
	
	

	
	
	
def main (): 
	data, att_data = parser("car.arff")
	strat_data = stratification(data)
	stratifiedCrossValidation(strat_data,att_data,10)

	estimator = Id3Estimator()

	test_paths = []#prepare test paths
	for i in range(10): test_paths.append("test"+str(i)+".arff")
	CV_learn(estimator,"train.arff",test_paths,10,att_data)
	
	data, att_data = parser("car.arff")

	
	
	option = "max_depth" 
	p_range = "1:1:10"
	params = option + "=" + "1"  


	tr_raw, tr_att = parser("train0.arff")
	te_raw, te_att = parser("train0.arff")

	tr_data, tr_target = splitData(tr_raw,tr_att)
	te_data, te_target = splitData(te_raw,te_att)

	classifier = Id3Estimator(params)
	classifier.fit(tr_data,tr_target, params)


	predictions = classifier.predict(te_data)
			
	ctr = 0
	for i in range(len(predictions)):
		if predictions[i] != te_target[i]:
			ctr+=1
	print((len(predictions)-ctr)/len(predictions))

	classifier = Id3Estimator(max_depth=1)
	classifier.fit(tr_data,tr_target )


	predictions = classifier.predict(te_data)
			
	ctr = 0
	for i in range(len(predictions)):
		if predictions[i] != te_target[i]:
			ctr+=1
	print((len(predictions)-ctr)/len(predictions))

Task1()













