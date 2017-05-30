import numpy as np
from sklearn.model_selection import train_test_split

def nonlin(x,deriv=False):			#sigmoid function	#for probabilities out of numbers
	if deriv==True:
		return x*(1-x)
	
	return 1/(1+np.exp(-x))

def scale_linear_bycolumn(rawpoints, high=1.0, low=0.0):		# To scale function between high and low
    mins = np.min(rawpoints, axis=0)
    maxs = np.max(rawpoints, axis=0)
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)

def trainNN(train_X,train_y,inp_nd=3,inter_node=5,iter=1000):
	#Synapses matrices  # Weight between connection		#connection between each neuron		#3 layer so 2 middle connections
	# Random initialisation
	syn0 = 2*np.random.random((inp_nd,inter_node)) - 1			#weight between input to hidden layer ("inp_node"- input node, "inter_node"- hidden layer node )
	syn1 = 2*np.random.random((inter_node,1)) - 1			# weight between hidden layer to output layer (restraint for 1 node output)
	
	#prev_error=100
	#training step
	for j in xrange(iter):
		l0 = train_X				#first layer input data
		l1 = nonlin(np.dot(l0,syn0))		#dot product between input and weight and their sigmoid function to predict prob for next layer
		l2 = nonlin(np.dot(l1,syn1))
		
		l2_error = train_y - l2			# Error of layer2 (Output)
		
		if(j % (iter/6))== 0:
			print "Error:" + str(np.mean(np.abs(l2_error)))		#average error at set interval so to monitor it goes down
		'''if np.mean(np.abs(l2_error)) == prev_error:
			print "not converging"
			break'''
		
		prev_error = np.mean(np.abs(l2_error))
		#multiply error with o/p of sigmoid function, which will give us delta which will improve the weight at every iteration
		#how much layer1 contributed to layer 2 layer2 delta multiply by layer1 weights
		
		l2_delta = l2_error*nonlin(l2,deriv=True)		#Delta of layer 2 wrt error
		
		l1_error = l2_delta.dot(syn1.T)					#Error of hidden layer wrt layer2 delta.
		
		l1_delta = l1_error*nonlin(l1,deriv=True)	#derivative of layer 1
		
		#Update weights
		syn1 += l1.T.dot(l2_delta)
		syn0 += l0.T.dot(l1_delta)
	return syn0,syn1
	
def predict(l0,syn0,syn1):						# Function to predict l0 <- x values , sync0,sync1 <- weight vectors for hidden layers
	l1 = nonlin(np.dot(l0,syn0))
	l2 = nonlin(np.dot(l1,syn1))
	return l2

if __name__ == "__main__":
	#input data as matrix		#each column is for 1 neuron
	inp = np.loadtxt(open("F:/idm/compressed/ANN-CI1/Diabetes.csv", "rb"), delimiter=",", skiprows=0)
	scale_inp = scale_linear_bycolumn(inp)		#Scale input data
	train, test = train_test_split(scale_inp, test_size = 0.25)		#split data into test and train
	
	train_X = np.array(train[:,0:8])
	train_y = np.array(train[:,8:9])
	
	test_X = np.array(test[:,0:8])
	test_y = np.array(test[:,8:9])
	
	"""
	#input data as matrix		#each column is for 1 neuron
	train_X = np.array([[0,0,1],
				[0,1,1],
				[1,0,1],
				[1,1,1]])
				
	#Output data		#4 o/p 1 output neuron each
	train_y= np.array([[6],
				[0],
				[1],
				[1]])
	
	test_X = np.array([[0,0,0],
						[1,0,0],
						[0,1,0]])
	test_y = np.array([[0],
						[1],
						[0]])
  """
	#count=0
	tot = len(test_y)
	np.random.seed(41)		#to make them deterministic
	
	for i in xrange(10,14):
		count=0
		syn0,syn1 = trainNN(train_X,train_y,8,i,6000)		# (input, output , input node, hidden layer node, # iterations)
		for idx,row in enumerate(test_X):
			#ans = int(round(predict(row,syn0,syn1)))
			ans = round(predict(row,syn0,syn1))
			orig = test_y[idx]
			#if ans[0]>ans[1]:
			#	a=0.
			#else:
			#	a=1.
			if ans==orig:
				count = count+1
			#print("row : {}:prediction is :{}, original value: {}".format(idx,ans,orig))
		print("NODE:{} - correct count : {}, out of {}".format(i,count,tot))
		print "accuracy:",int(count/tot)
		
	#syn0,syn1 = trainNN(train_X,train_y,11,60000)
	#print "predict for new test set test_X"
	
	#for idx,row in enumerate(test_X):
	#	ans = int(round(predict(row,syn0,syn1)))
	#	orig = test_y[idx]
	#	print("row : {}:prediction is :{}, original value: {}".format(idx,ans,orig))
	#	if ans==orig:
	#		count = count+1
	
	#print("correct count : {}, out of {}".format(count,tot))
	#acc=(int(count)/int(tot))*100
	#print "Accuracy is :",acc
