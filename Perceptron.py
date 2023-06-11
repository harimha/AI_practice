import matplotlib.pyplot as plt
import numpy as np

class Perceptron():
    def __init__(self, nDim, w=[], b=0):
        if len(w) > 0:
            self.w = np.array(w)
            self.w0 = b
        else:
            w = np.random.rand(nDim+1)
            self.w0 = w[0]
            self.w = w[1:]
            
    def printW(self):
        '''가중치 출력 메서드'''
        f = '  w{} = {:6.3f}'
        print(f.format(0, self.w0), end='')
        for i in range(self.nDim):
            print(f.format(i+1, self.w[i]), end='')
        print()
        
    def predict(self, x):
        return int(np.dot(self.w, x) + self.w0 >= 0) 
    
    def fit(self, x, y, n, ephocs, eta=0.01):
        f = 'Epochs = {:4d}'
        print(f.format(0), end='')
        self.printW()
        
        for j in range(1, ephocs+1):
            flag = True # weight에 변화가 없는 경우 학습 중지를 위함
            for i in range(n):
                value = self.predict(x[i])
                err = value - y[i]
                if err != 0 :
                    self.w -= eta*err*x[i]
                    self.w0 -= eta*err
                    flag = False
                    print(f.format(j), end='')
                    self.printW()
                    if flag : break
           
# 초기값
nSamples= 10
nDim = 2
nEpochs = 1000
eta = 0.1

# 학습 표본
x = np.array([[0.25, 0.75], 
              [1.25, 1.75],
              [0.5, 0.5], 
              [1.75, 1.25],
              [0.75, 0.25], 
              [1.5, 1.5],
              [0.0, 1.5], 
              [2.5, 0.0],
              [1.5, 0.0], 
              [0.0, 2.5]])
y = np.array([1,0,1,0,1,0,1,0,1,0])

# weight 초기값
w = np.array([0.1, -0.3]) # w1 , w2
b = -1 # w0

p = Perceptron(nDim, w, b)
p.fit(x, y, nSamples, nEpochs, eta)

# 결과 시각화
nGrid = 61
x1_lin = np.linspace(0, 3, nGrid)
x2_lin = np.linspace(0, 3, nGrid)

x1_mesh, x2_mesh = np.meshgrid(x2_lin, x1_lin)

xx = np.vstack([x1_mesh.ravel(), x2_mesh.ravel()]).T



