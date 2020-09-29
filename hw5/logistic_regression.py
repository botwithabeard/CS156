import numpy as np

def check(xt,yt):
    x0,y0,x1,y1 = np.random.uniform(-1,1,4)
    m = (y1-y0)/(x1-x0)
    return np.sign(m*(xt-x0)-(yt-y0))

def generatedataset(numberOfTrials=100):
    dataset = np.ones((numberOfTrials,4))

    for i in range(numberOfTrials):
        xt,yt = np.random.uniform(-1,1,2)
        dataset[i,1:] = xt,yt,check(xt,yt)

    return dataset

def sigmoid(data, weight):
    theta = np.dot(data,weight.T)
    return 1/(1+np.exp(-theta))

def loss(hypo, y):
    return (-y * np.log(hypo) - (1 - y) * np.log(1 - hypo)).mean()

def gradient_descent(data, h, y):
    return np.dot(data.T, (h - y)) / y.shape[0]

def update_weight_loss(weight, gradient):
    learning_rate=0.1
    return weight - learning_rate * gradient

def grad(data):
    rows,cols = data.shape
    x = data[:,:cols-1]
    y = data[:,cols-1]
    w = np.random.rand(cols-1)
    i= 0
    while(loss(sigmoid(x,w),y)>0.0001):
        i+=1
        h = loss(sigmoid(x,w),y)
        grad = gradient_descent(x,h,y)
        w = update_weight_loss(w,grad)

    error = np.mean(sigmoid(x,w))


    return w,error,i

if __name__ == '__main__':
    data = generatedataset(numberOfTrials = 100)
    print('this is dataset')
    print(data)
    print('---------------')
    print(grad(data))
