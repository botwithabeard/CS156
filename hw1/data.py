import numpy as np
import random

def check(xt,yt):
    x0,y0,x1,y1 = np.random.uniform(-1,1,4)
    m = (y1-y0)/(x1-x0)
    return np.sign(m*(xt-x0)-(yt-y0))

def generatedataset(numberOfTrials):
    dataset = []
    result = np.zeros([numberOfTrials,1])

    for i in range(numberOfTrials):
        xt,yt = np.random.uniform(-1,1,2)
        dataset.append([xt,yt])
        result[i,0] = check(xt,yt)

    return np.array(dataset), result

def perceptron(data,result):
    w = np.random.rand(1,3)
    x = np.ones((data.shape[0],data.shape[1]+1))
    final = np.ones(result.shape)
    counter = 0
    x[:,1:] = data # this is the final dataset w w0 adjusted
    iterations =  0
    while(np.any(final!=result)):
        iterations+=1
        idx = random.choice(range(data.shape[0]))
        vector  = x[idx]
        if(np.sum(w*vector) > 0):
            if(result[idx]==-1):
                print('misclassified case 1')
                w+=(1-w*vector)*vector
                final[idx] = -1
        if(np.sum(w*vector) <= 0):
            if(result[idx] ==  1):
                print('misclassified case 2')
                w+=(1+w*vector)*vector
                final[idx] = 1
    return -w,iterations

if __name__ == '__main__':
    data, result = generatedataset(10)
    print(data)
    print(result)
    w,iterations= perceptron(data,result)
    print(w)
    print(iterations)
    print()
