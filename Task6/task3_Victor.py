import numpy as np
class perceptron:
    def __init__(self,Weights): #len of weights corresponds to input +1 for bias
        self.weights = Weights
    def calc(self,inp):
        if len(inp)+1 != len(self.weights):
            print("input does not match weights dimensions")
            return 0
        else:
            out = 0
            for i in range(len(inp)):
                out+=inp[i]*self.weights[i]
            out+=self.weights[len(self.weights)-1]

            if out < 1: out = 0
            else: out = 1
            return out

class layer:
    def __init__(self,perc):
        self.perceptrons = perc
    def calc(self,inp): #calc perc value for each and return arr
        out = []
        for i in range(len(self.perceptrons)):
            out.append(self.perceptrons[i].calc(inp))
        return out

class MLP:
    def __init__(self,lyrs):
        self.layers = lyrs
    def calc(self,inp):
        if len(self.layers) == 0:
            print("no layers")
            return 0
        else:
            out = self.layers[0].calc(inp)
            for i in range(len(self.layers)-1):
                out = self.layers[i+1].calc(out)

            out = np.sum(out)
            return out

def testOutcomes(mlperc):
    print("0,0 :" + str(mlperc.calc([0, 0])))
    print("0,1 :" + str(mlperc.calc([0, 1])))
    print("1,0 :" + str(mlperc.calc([1, 0])))
    print("1,1 :" + str(mlperc.calc([1, 1])))


p1 = perceptron([1,1,0.5])#or gate
p2 = perceptron([1,1,-0.5])#and gate
p3 = perceptron([1,-1,0.5])#or gate mit not-2. input  (and)  XOR = OR(OR(X1,X2),not(AND(X1,X2)))


l1 = layer([p1,p2])
l2 = layer([p3])

XOR = MLP([l1,l2])

testOutcomes(XOR)

#todo: sketch decision boundaries