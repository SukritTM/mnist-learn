import numpy as np
from typing import List
import random

class Layer:
	# implements a layer of neurons and weights, activation function is sigmoid
	def __init__(self, size: int, final: bool = False):
		self.size = size
		self.final = final

		self.activations = np.zeros(self.size) # dim: (1, size)
		
		if not final:
			self.weights = None                # dim: (size, nxtsize)
			self.biases = None                 # dim: (1, nxtsize)
	
	def init_weights_biases(self, nxtsize):
		# initialise weights and biases seperately, because we need the next layer to know what the
		# dimensions of this have to be
		if self.final:
			raise AttributeError('Attempted initialising paramaters in a final layer')

		# TODO: Make this random array better with random module 
		self.weights = np.random.randn(self.size, nxtsize)
		self.biases = np.random.randn(nxtsize)

		
	
	def feedforward(self):
		# returns the weight computation of the layer, feed this into the next 
		if self.final:
			raise AttributeError('Cannot feed forward from a final layer')

		return self.activations@self.weights + self.biases

class Network:
	def __init__(self, *layers: Layer):
		self.layers = layers
		self.layers[-1].final = True

		for i in range(len(self.layers)-1):
			self.layers[i].init_weights_biases(self.layers[i+1].size)
		
	def predict(self):
		for i in range(len(self.layers)-1):
			next = self.layers[i].feedforward()
			assert next.shape == self.layers[i+1].activations.shape
			self.layers[i+1].activations = next

		return self.layers[-1].activations
		