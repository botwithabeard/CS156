import numpy as np
import random

def flip(flips):
    result = []
    for i in range(flips):
        result.append(random.choice([1,0]))
    return result

def freqcount(coin):
    return float(np.sum(coin))/float(len(coin))

def simulate(coins=1000, flips=10):
    results = []
    for coin in range(coins):
        results.append(flip(flips))

    c1 = np.array(results[0])
    crand = np.array(random.choice(results))
    cmin = np.array(min(results))

    v1 = freqcount(c1)
    vrand = freqcount(crand)
    vmin = freqcount(cmin)

    return v1,vrand,vmin

def check(xt,yt):
    x0,y0,x1,y1 = np.random.uniform(-1,1,4)
    m = (y1-y0)/(x1-x0)
    return np.sign(m*(xt-x0)-(yt-y0))

def generatedataset(numberOfTrials):
    dataset = np.zeros((numberOfTrials,3))

    for i in range(numberOfTrials):
        xt,yt = np.random.uniform(-1,1,2)
        dataset[i,:] = xt,yt,check(xt,yt)

    return dataset

def linear_regression(dataset):
    points, parameters = dataset.shape
    X = dataset[:,:(parameters-1)]
    H = np.linalg.inv((X.T).dot(X)).dot(X.T)
    W = H.dot(dataset[:,(parameters-1)])

    return W

def value_g(trials = 1000):
    results = np.ones((trials,4))

    for i in range(trials):
        data = generatedataset(100)
        out = np.sign(np.sum(linear_regression(data)*data[:,:3], axis=1))
        correct = (out==data[:,3])
        results[i,1:] = (float(np.sum(out==True))/float(100))

    return results

def newdataset(points=1000):
    data = np.ones((points,7))

    for i in range(points):
        x1,x2 = np.random.uniform(-1,1,2)
        data[i,1:] = x1,x2,x1*x2,x1**2,x2**2, np.sign(x1**2+x2**2-0.6)

    #  addding noise
    idx = np.random.choice(data.shape[0], 100, replace=False)
    for i in idx:
        data[i,5] = data[i,3]*-1


    return data

def value_h(trials = 1000):
    results = np.ones(trials)
    w_av = np.zeros((trials,6))

    for i in range(trials):
        data = newdataset()
        w = linear_regression(data)
        w_av[i,:] = w
        out = np.sign(np.sum(w*data[:,:6], axis=1))
        correct = (out==data[:,6])
        results[i] = (np.sum(correct==True))/float(1000)

    print(results)
    print(np.mean(results))
    print(np.mean(w_av,axis=0))
    return results

if __name__ == '__main__':
    # print(simulate())
    #
    # # linear regression
    result = value_h()
    #
    # sample = random.sample(list(result),10)
    #
    print(float(np.sum(result))/float(len(result)))
