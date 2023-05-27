import numpy as np

class LinRegression():
    def __init__(self, nDim, w=[]):
        self.nDim = nDim
        if len(w) > 0:
            self.w = np.array(w)
        else:
            self.w = np.random.rand(nDim+1)

    def fit(self, x, y, N, ephocs, eta=0.01):
        cost = 0
        for i in range(N):
            value = np.dot(self.w, x[i])
            err = value - y[i]
            cost += err * err
        cost /= N
        print("Epochs = {:4d}".format(0), end=" ")
        for i in range(self.nDim+1):
            print("  w{} = {:.3f}".format(i, self.w[i]), end=" ")
        print("  cost = {:.3f}".format(cost))

        for j in range(1, ephocs+1):
            dw = np.zeros(self.nDim+1)
            for i in range(N):
                value = np.dot(self.w, x[i])
                err = value - y[i]
                dw += err * x[i]

            self.w -= eta * 2.0 *dw/N
            if j % 20 == 0 :
                for i in range(N):
                    value = np.dot(self.w, x[i])
                    err = value - y[i]
                    cost += err * err
                cost /= N
                print("Epochs = {:4d}".format(j), end=" ")
                for i in range(self.nDim + 1):
                    print("  w{} = {:.3f}".format(i, self.w[i]), end=" ")
                print("  cost = {:.3f}".format(cost))

    def predict(self, x):
        return np.dot(self.w, x)

nSamples = 5
nDim = 1
nEpochs = 3000
eta = 0.01

x = np.array([[1,10], [1,9], [1,6], [1,4], [1,2]])
y = np.array([93,80,77,60,30])

w = np.array([0.3, 0.5])

linR = LinRegression(1,w)
linR.fit(x, y, nSamples, nEpochs, eta)


xx = np.array([1, 5])
print("")
print("x = {:.0f} --> y = {:.1f}". format(xx[1], linR.predict(xx)))
xx = np.array([1, 8])
print("x = {:.0f} --> y = {:.1f}". format(xx[1], linR.predict(xx)))


