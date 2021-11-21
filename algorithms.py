import csv
import math
import sys

# Algorithm 1
def calculate_mec(data, labels):
	'''
	Calculate MEC needed for a classifier assuming weight
	equilibrium (no gradient descent). 
	Requires: data array of length n  which contains d-dim vectors 
	x, and a label column of length n.

	Parameters
	----------
	data : array
	
	Returns
	-------
	mec : float

	'''
	assert len(data) == len(labels)

	thresholds = 0
	table = []
	for i in range(len(data)):
	  # Sum up all data in the row and append the label
	  table.append([data[i].sum(), labels[i]])
	# Sort by sum in reverse order
	sortedtable = sorted(table, key=lambda x: x[0], reverse=True)

	# Iterate through sorted table and compute thresholds
	# by comparing class labels for each row and the prior row
	# Initialize with first class label in list
	class_label = sortedtable[0][1]
	for i in range(len(sortedtable)):
	  if not sortedtable[i][1] == class_label:
	    thresholds = thresholds + 1 
	    class_label = sortedtable[i][1]

	mec = math.log2(thresholds + 1)
	return mec

def get_sample(p, data, labels):
	'''
	Return p percent of the data with corresponding labels.
	'''
	num_labels = len(labels)

	size = p * num_labels
	
	np.random.shuffle(data)
	np.random.shuffle(labels)
	
	return data[:size], labels[:size] 



# Algorithm 2 
def calculate_capacity_progression(data, labels):
	'''
	Calculate capacity progression for the Equilibrium Machine Learner.
	Requires: data array of length n  which contains d-dim vectors 
	x, and a binary label column of length n.

	Parameters
	----------
	data : array
	labels : array
	
	Returns
	-------
	mec : float

	'''
	sizes = [.05, .1, .2, .4, .8, 1]
	for size in sizes:
		sample_data, sample_labels = get_sample(size, data, labels)
		mec = calculate_mec(sample_data, sample_labels)
	return "MEC for " + size + "percent of the data: " + mec + " bits"
