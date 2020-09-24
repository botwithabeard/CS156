import numpy as np
import random

def getPoints():
    """
    Gives random points from the curve
    """
    TOTAL = 2
    points = np.zeros((TOTAL,TOTAL))
    for i in range(TOTAL):
        x = np.random.choice((np.arange(-1,1,0.1,dtype='float16')))
        y = np.sin(np.pi*x)
        points[i] = x,y

    return points

def getSlope(points):
    return (points[1,1]-points[0,1])/(np.logaddexp(points[1,0],-points[0,0]))

def getAverage(numberOfTrials = 1000):
    out = np.zeros(numberOfTrials)
    for i in range(numberOfTrials):
        out[i] = getSlope(getPoints())

    return np.mean(out)

if __name__ == '__main__':
    print(getAverage(numberOfTrials = 100000))
