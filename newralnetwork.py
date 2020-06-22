import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

def step_function_np(x):
    y = x > 0
    return y.astype(np.int)

def step_function_2(x):
    return np.array(x > 0, dtype=np.int)

#x = np.arange(-5.0, 5.0, 0.1)
#y = step_function_2(x)
#plt.plot(x, y)
#plt.ylim(-0.1, 1.1)
#plt.show()


def sigmoid(x):
    return 1/(1 + np.exp(-x))

#x = np.arange(-5.0, 5.0, 0.1)
#y = sigmoid(x)
#plt.plot(x, y)
#plt.ylim(-0.1, 1.1)
#plt.show()


def relu(x):
    return np.maximum(0, x)

#x = np.array([1.0, 0.5])
#w1 = np.array([0.1, 0.3, 0.5], [0.2, 0.4, 0.6])
#b1 = np.array([0.1, 0.2, 0.3])
#
#print(w1.shape)
#print(x.shape)
#print(b1.shape)
#
#a1 = np.dot(x, w1) + b1
#z1 = sigmoid(a1)
#
#w2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
#b2 = ([0.1, 0.2])
#
#a2 = np.dot(z1, w2) + b2
#z2 = sigmoid(a2)
#
#w3 = np.array([[0.1, 0.3], [0.2, 0.4]])
#b3 = np.array([0.1, 0.2])
#
#a3 = np.dot(z2, w3) + b3

def identity_function(x):
    return x

#z3 = identity_function(a3)



#ソフトマックス関数
def softmax(x):
    c = np.max(x)
    exp_a = np.exp(x - c) #p69参照
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

