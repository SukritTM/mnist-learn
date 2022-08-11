from layers import Layer, Network
import numpy as np

model = Network(Layer(5, activation='relu'),
				Layer(1, activation='sigmoid'),
				
				input_size = 1)

in_set = np.asarray([[1], [3], [10], [100], [5], [250], [322], [12], [144], [289]])
lab_set = np.asarray([[0], [0], [0], [1], [0], [1], [1], [0], [1], [1]])

model.train(in_set, lab_set, 100, alpha=0.1)

print(model.predict(np.asarray([15])))
print(model.predict(np.asarray([250])))
print(model.predict(np.asarray([60])))
print(model.predict(np.asarray([90])))
print(model.predict(np.asarray([99])))
print(model.predict(np.asarray([100])))