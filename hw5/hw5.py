import numpy as np
import math


def func(u,v):
	return (u*math.exp(v)-2*v*math.exp(-u))**2

def grad_u(u,v):
	return 2*(u*math.exp(v)-2*v*math.exp(-u))*(math.exp(v)+2*v*math.exp(-u))

def grad_v(u,v):
	return 2*(u*math.exp(v)-2*v*math.exp(-u))*(u*math.exp(v)-2*math.exp(-u))

def simple_grad_descent(u=1,v=1,learning_rate=0.1):
    error = func(u,v)
    iterations = 0

    while(error > 10**(-14)):
        iterations+=1
        du = grad_u(u,v)
        dv = grad_v(u,v)
        u = u - du*learning_rate
        v = v - dv*learning_rate
        error = func(u,v)
        # print(error)

    return iterations,u,v

def coordinate_grad_descent(u=1,v=1,learning_rate=0.1,steps = 30):
    error = func(u,v)
    iterations = 0

    while(iterations < steps):
        iterations+=1
        du = grad_u(u,v)
        u = u - du*learning_rate
        error = func(u,v)
        dv = grad_v(u,v)
        v = v - dv*learning_rate
        error = func(u,v)
        # print(error)

    return iterations,u,v





if __name__ == '__main__':
    print(simple_grad_descent())
    print(coordinate_grad_descent())
