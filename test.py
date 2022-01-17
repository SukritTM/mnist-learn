from utils import Layer, Network

model = Network(Layer(10),
				Layer(5),
				Layer(2))

print(model.predict())