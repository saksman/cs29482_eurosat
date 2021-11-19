import csv
import math
import sys

# y:u:v 4:1:1

# Algorithm 1
def calculate_mec(data, labels):
	'''
	Calculate MEC needed for a binary classifier assuming weight
	equilibrium (no gradient descent). 
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
	assert len(data) == len(labels)

	thresholds = 0
	table = []
	for i in range(len(data)):
		# Sum up all data in the row and append the label
		# What format should the data be in???
		table.append(data[i].sum(), label[i])
	sortedtable = sorted(table, key=lambda x: x[0], reverse=True)
	label_class = 0

# 1, 1, 1, 1, 2, 2, 2, 3
# for each class change, add threshold,
# class change, not 0 is for comparison 

	for i in range(len(sortedtable)):
		if not sortedtable[i][1] == label_class:
			label_class = sortedtable[i][1]
			thresholds = thresholds + 1 

	mec = math.log2(thresholds + 1)
	return mec

# What is i? What is j?
# Can we update our project to be about a balanced subset of our dataset?

def getSample(p, labels):
	'''
	Return p percent of the data with corresponding labels.
	'''
	num_labels = len(labels)

	subset_labels = p * num_labels
	np.random.shuffle(labels)
	
	return labels[:subset_labels] 



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
	sizes = [5, 10, 20, 40, 80, 100]
	for size in sizes:
		subset = getSample(size, labels)
		mec = calculate_mec(subset)
	return "MEC for " + size + "percent of the data: " + mec + " bits"
