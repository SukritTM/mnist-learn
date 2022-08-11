import numpy as np
from typing import List, Tuple
import random

class Layer:
	# implements a layer of neurons and weights, activation function is sigmoid
	def __init__(self, size: int, initial: bool = False, final: bool = False, activation=None):
		self.size = size
		self.final = final
		self.activation = activation
		self.inputs = None
		self.input_size = None
		self.curr_out = None
		
		if not initial:
			self.weights = None                     # dim: (size, prevsize)
			self.biases = None                      # dim: (size, 1)
	
	def init_weights_biases(self, prevsize: int):
		# initialise weights and biases seperately, because we need this layer to know what the
		# dimensions of the previous have to be

		# TODO: Make this random array better with random module 
		self.input_size = prevsize
		self.weights = np.random.randn(self.size, prevsize)
		self.biases = np.random.randn(self.size, 1)		
	
	def feedforward(self, inputs):
		# returns the weight computation of the layer, feed this into the next

		self.inputs = inputs
		y = self.weights@inputs + self.biases 
		self.curr_out = y

		if self.activation == 'relu':
			return y * (y>0)

		if self.activation == 'sigmoid':
			return 1/(1 + np.exp(-y))

		return y

	def backprop(self, dE_dY, alpha: int):


		if self.activation == 'sigmoid':
			sig = lambda x: 1/(1 + np.exp(-x))
			dE_dY = sig(self.curr_out)*(np.ones_like(self.curr_out)-sig(self.curr_out))*dE_dY

		if self.activation == 'relu':
			relu_p = lambda x: 1 * (x>0)
			dE_dY = relu_p(self.curr_out)*dE_dY
		

		dE_dW = dE_dY@self.inputs.reshape(1, self.input_size)
		dE_dB = dE_dY
		dE_dX = np.transpose(self.weights)@dE_dY

		self.weights -= alpha*dE_dW
		self.biases -= alpha+dE_dB

		return dE_dX


class Network:
	def __init__(self, *layers: Layer, input_size: int):
		self.layers = layers
		self.layers[-1].final = True
		self.layers[0].initial = True
		self.input_size = input_size

		self.layers[0].init_weights_biases(input_size)
		for i in range(1, len(self.layers)):
			self.layers[i].init_weights_biases(self.layers[i-1].size)
		
	def predict(self, data):
		data = data.reshape(self.input_size, 1)
		for i in range(len(self.layers)):
			data = self.layers[i].feedforward(data)

		return data
	
	def train(self, inputs, labels, epochs, alpha=0.01, label_mode='dense'):
		for i in range(epochs):
			print(f'Epoch {i+1}/{epochs}', end='')
			cost = self.train_single(inputs, labels, alpha)
			print(f' - cost: {cost}')

	def train_single(self, input_batch, label_batch, alpha):
		cost = 0
		cost_p = np.zeros((self.layers[-1].size, 1))
		# print(cost_p.shape)
		for i in range(len(input_batch)):
			pred = self.predict(input_batch[i])
			label = label_batch[i]
			arr = np.zeros((self.layers[-1].size, 1))

			if arr.shape[0] == 1:
				arr[0, 0] = label
			else:
				arr[label, 0] = 1
			label = arr

			cost += np.sum(np.power(pred-label, 2))
			cost_p = cost_p + 2*(pred-label)
		
		cost_p /= len(input_batch)
		cost /= len(input_batch)

		for i in range(len(self.layers)-1, -1, -1):
			cost_p = self.layers[i].backprop(cost_p, alpha)

		return cost