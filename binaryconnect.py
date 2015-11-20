from blocks.bricks import Linear, Softmax, Logistic, MLP, Rectifier
from blocks.bricks import application
from blocks.bricks.conv import *
from blocks.algorithms import GradientDescent, Scale
from blocks.bricks.cost import CategoricalCrossEntropy, Cost
from blocks.initialization import IsotropicGaussian
from blocks.graph import ComputationalGraph
from blocks.main_lopp import MainLoop
from blocks.extensions import SimpleExtension
from blocks.initialization import *
import theano.tensor as T

def hard_sigmoid(x):
	return T.max(0, T.min(1, x + 1/2))



class Update(SimpleExtension):
	''' Correctly update weights
	'''
	pass

class MLPBinarize:
	def __init__(self, Xdata, ydata):
		self.Xdata = Xdata
		self.ydata = ydata

	def _binarize(self, W):
		prob = hard_sigmoid(W)
		return 1 if prob > 0 else -1

	def run(self, rate=.001):
		x = T.matrix('features')
		y = T.lmatrix('targets')
		loss = CategoricalCrossEntropy().apply(y.flatten(), output)
		gr = ComputationalGraph(loss)
		monitor = DataStreamMonitoring(variables=[loss], data_stream=test_set_monitor())
		algorithm = GradientDescent(cost=loss, step_rule=Scale(learning_rate=lrate), params=gr.parameters)
		loop = MainLoop(data_stream=data_stream, algorithm=algorithm, extensions=[monitor, Printing()])
		loop.run()