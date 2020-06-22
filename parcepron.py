import numpy as np

def AND(x1, x2):
    W1, W2, theta = (0.5, 0.5, 0.7)

    temp = x1 * W1 + x2 * W2
    if temp <= theta:
        return 0

    elif temp > theta:
        return 1

print(AND(0, 0))
print(AND(1, 0))
print(AND(0, 1))
print(AND(1, 1))
print("\n")


def AND1(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    y = np.sum(x * w) + b

    if y > 0:
        return 1
    else:
        return 0

print(AND1(0, 0))
print(AND1(1, 0))
print(AND1(0, 1))
print(AND1(1, 1))
print("\n")


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    y = np.sum(x * w) + b

    if y > 0:
        return 1
    else:
        return 0

print(NAND(0, 0))
print(NAND(1, 0))
print(NAND(0, 1))
print(NAND(1, 1))
print("\n")


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    y = np.sum(x * w) + b

    if y > 0:
        return 1
    else:
        return 0

print(OR(0, 0))
print(OR(1, 0))
print(OR(0, 1))
print(OR(1, 1))
print("\n")


def XOR(x1, x2):
    s1 = NAND(x1 ,x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

print(XOR(0, 0))
print(XOR(1, 0))
print(XOR(0, 1))
print(XOR(1, 1))
print("\n")
