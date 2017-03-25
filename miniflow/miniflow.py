''' In this implementation during forward propogation the matrices are multiplied as [X*W), but not as in a conventional way of [W*X]
'''

import numpy as np

class Node(object):
	def __init__(self,inbound_nodes=[]):
		self.inbound_nodes = inbound_nodes
		# every node has a value, so initialize it to None
		self.value = None
		#every node can propogate its output along the graph to any number of nodes , So initialize with a list
		self.outbound_nodes = []
		# when we are moving along the graph we donot know the nodes that may appear beyond our node but have a specific set of input nodes. So the outbound nodes is written in this way : indicate the current node as ouput node to all the input nodes
		for node in inbound_nodes:
			node.outbound_nodes.append(self)# append the current node to the list of ouput nodes that input is connected to
		#To find decrease the amount of loss we need to back propogate and for that we need gradients, but Gradients are not list and is a dictionary with node as key and gradint as value
		self.gradients={}
	# every node requires two methods 1) Forward , 2) Backward , They are initializes here but the value through them shall be calculated depending on the characteristics of the node
	def forward(self):
		raise NotImplementedError

	def Backward(self):
		raise NotImplementedError
	# We specially define a node that has no inputs Which is the INPUT node of the entire network
class Input(Node):
	def __init__(self):
		Node.__init__(self)# we donot have the param inbound_nodes

	def forward(self):
		pass# pass as we donot calculate anything at this point
	def Backward(self):
		#The gradient is 0 w.r.t no inbound_nodes but this node itself has oubound nodes, and the gradient is generally calculated when the process is back propogating
		self.gradients={self:0}
		for node in outbound_nodes:
			self.gradients[self] += node.gradients[self]
			# The gradients are added at the self node as it can be the input of any number of nodes. So, we pick the gradients from those output nodes and sum them up.
class Linear(Node):
	def __init__(self,X,W,b):
		Node.__init__(self,[X,W,b])# the inbound nodes to the calculation class linear are X:input, W:weights, b:bias.
	def forward(self):
		X=self.inbound_nodes[0].value# In these three cases X,W,b should be called as Input() methods, value to those input nodes is assigned when they a re passed through the topological sort 
		W=self.inbound_nodes[1].value# a method that is implemented ahead.
		b=self.inbound_nodes[2].value
		self.value = np.dot(X,W)+b
	def Backward(Node):
		# back prop is implemented from this node to the inputs, So initialize gradient with a size of input node
		self.gradients = {n:np.zeroes_like(n.value) for n in self.inbound_nodes}
		for n in self.outbound_nodes:
			# we need to quantify the cost of the node
			grad = n.gradients[self]
			self.gradients[self.inbound_nodes[0]] += np.dot(grad,self.outbound_nodes[1].value.T)
			self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T,grad)
			self.gradients[self.inbound_nodes[2]] += np.sum(grad, axis=0, keepdims=False)

class Sigmoid(Node):
	def __init__(self,node):
		Node.__init__(self,[node])
	def sigmoid(self,x):
		return (1/(1+np.exp(-x)))
	def forward(self):
		self.value=self.sigmoid(self.inbound_nodes[0].value)
	def Backward(self):
		self.gradients= {n:np.zeroes_like(n.value) for n in self.inbound_nodes}
		for n in self.outbound_nodes:
			grad = n.gradients[self]
			sigma = self.value
			self.gradients[self.inbound_nodes[0]] += sigma*(1-sigma)*grad
class MeanSqErr(Node):
	def __init__(self,y,a):
		Node.__init__(self,[y,a])
	def forward(self):
		# Making both arrays (3,1) insures the result is (3,1) and does
        # an elementwise subtraction as expected.
		y= self.inbound_nodes[0].values.reshape(-1,1)
		a= self.inbound_nodes[1].values.reshape(-1,1)
		self.m = self.inbound_nodes[0].value.shape[0]
		# Save the computed output for backward as self.m
		self.diff = y-a
		self.value = np.mean((y-a)**2)
	def Backward(self):
		self.gradients[self.inbound_nodes[0]]= (2/self.m)*self.diff
		self.gradients[self.inbound_nodes[1]]= (-2/self.m)*self.diff
def topological_sort(feed_dict):
    """
    Sort the nodes in topological order using Kahn's Algorithm.
    `feed_dict`: A dictionary where the key is a `Input` Node and the value is the respective value feed to that Node.
    Returns a list of sorted nodes.
    """
    input_nodes = [n for n in feed_dict.keys()]
    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_and_backward(graph):
    """
    Performs a forward pass and a backward pass through a list of sorted Nodes. Arguments: `graph`: The result of calling `topological_sort`.
    """
    # Forward pass
    for n in graph:
        n.forward()

    # Backward pass
    # see: https://docs.python.org/2.3/whatsnew/section-slices.html
    for n in graph[::-1]:
        n.backward()


def sgd_update(trainables, learning_rate=1e-2):
    """
    Updates the value of each trainable with SGD.

    Arguments:

        `trainables`: A list of `Input` Nodes representing weights/biases.
        `learning_rate`: The learning rate.
    """
    # Performs SGD
    #
    # Loop over the trainables
    for t in trainables:
        # Change the trainable's value by subtracting the learning rate
        # multiplied by the partial of the cost with respect to this
        # trainable.
        partial = t.gradients[t]
        t.value -= learning_rate * partial
