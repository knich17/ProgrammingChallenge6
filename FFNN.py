from data_parser import *
from math import *

class neural_network:
	# Gradient descent parameters (I picked these by hand)
	#epsilon = 0.01 # learning rate for gradient descent
	#reg_lambda = 0.01 # regularization strength
	def __init__(self, num_examples, in_dim, out_dim, epsilon, reg_lambda, h_dim, X, y):
		self.num_examples = num_examples
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.epsilon = epsilon
		self.reg_lambda = reg_lambda
		self.X = X
		self.y = y
		self.model = {}

		self.build_model(h_dim, 200000, True)

	def calculate_loss(self):
		W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
		# Forward propagation to calculate our predictions
		z1 = self.X.dot(W1) + b1
		a1 = np.tanh(z1)
		z2 = a1.dot(W2) + b2
		exp_scores = np.exp(z2)
		probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
		# Calculating the loss
		corect_logprobs = -np.log(probs[range(self.num_examples), self.y])
		data_loss = np.sum(corect_logprobs)
		# Add regulatization term to loss (optional)
		data_loss += self.reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
		return 1./self.num_examples * data_loss

	# Helper function to predict an output (0 or 1)
	def predict(self, x):
		W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
		# Forward propagation
		z1 = x.dot(W1) + b1
		a1 = np.tanh(z1)
		z2 = a1.dot(W2) + b2
		exp_scores = np.exp(z2)
		probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
		return np.argmax(probs, axis=1)

	# This function learns parameters for the neural network and returns the self.model.
	# - nn_hdim: Number of nodes in the hidden layer
	# - num_passes: Number of passes through the training data for gradient descent
	# - print_loss: If True, print the loss every 1000 iterations
	def build_model(self, nn_hdim, num_passes=20000, print_loss=False):
		 
		# Initialize the parameters to random values. We need to learn these.
		np.random.seed(0)
		W1 = np.random.randn(self.in_dim, nn_hdim) / np.sqrt(self.in_dim)
		b1 = np.zeros((1, nn_hdim))
		W2 = np.random.randn(nn_hdim, self.out_dim) / np.sqrt(nn_hdim)
		b2 = np.zeros((1, self.out_dim))
		 
		# Gradient descent. For each batch...
		for i in xrange(0, num_passes):

			# Forward propagation
			z1 = self.X.dot(W1) + b1
			a1 = np.tanh(z1)
			z2 = a1.dot(W2) + b2
			exp_scores = np.exp(z2)
			probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

			# Backpropagation
			delta3 = probs
			delta3[range(self.num_examples), self.y] -= 1
			dW2 = (a1.T).dot(delta3)
			db2 = np.sum(delta3, axis=0, keepdims=True)
			delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
			dW1 = np.dot(self.X.T, delta2)
			db1 = np.sum(delta2, axis=0)

			# Add regularization terms (b1 and b2 don't have regularization terms)
			dW2 += self.reg_lambda * W2
			dW1 += self.reg_lambda * W1

			# Gradient descent parameter update
			W1 += -self.epsilon * dW1
			b1 += -self.epsilon * db1
			W2 += -self.epsilon * dW2
			b2 += -self.epsilon * db2
			 
			# Assign new parameters to the self.model
			self.model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
			 
			# Optionally print the loss.
			# This is expensive because it uses the whole dataset, so we don't want to do it too often.
			if print_loss and i % 1000 == 0:
				loss = self.calculate_loss()
				print "Loss after iteration %i: %f" %(i, loss)


if __name__ == "__main__":
	in_dim = 15
	out_dim = 2
	# swap between the two lines for larger/smaller array printing
	np.set_printoptions(suppress=True) # print small
	# np.set_printoptions(threshold='nan', precision=4, suppress=True, linewidth=200) # print large

	parsed_data = dataParser()
	train_data, test_data = shuffleNSplit(parsed_data.data)
	solutions = train_data[:, -1].astype(int)
	train_data = train_data[:, 0:-1]

	print type(solutions)

	nn = neural_network(len(train_data), len(train_data[0]), out_dim, 0.01, 0.01, 3, train_data, solutions)

	totalCorrect = 0
	for test in test_data:
		print test[-1], nn.predict(test[:-1])[0]
		if (test[-1] == nn.predict(test[:-1])[0]):
			totalCorrect += 1
	print totalCorrect, len(test_data)

#http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/