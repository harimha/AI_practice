import matplotlib.pyplot as plt
import numpy as np

class Perceptron():
    def __init__(self, nDim, w=[], b=0):
        self.nDim = nDim
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
x1_lin = np.linspace(0, 3, nGrid) # 0 ~ 3 까지 61개로 동일간격으로 나눔
x2_lin = np.linspace(0, 3, nGrid)

x1_mesh, x2_mesh = np.meshgrid(x2_lin, x1_lin) # 2차원 공간 생성

xx = np.vstack([x1_mesh.ravel(), x2_mesh.ravel()]).T

# 학습된 Perceptron으로 xx에 대한 출력 계산
yy = np.empty(nGrid*nGrid, dtype=int)
for k in range(nGrid*nGrid):
    yy[k] = p.predict(xx[k])

# 출력할 그래프의 크기 및 좌표 범위 설정
plt.rcParams["figure.figsize"] = (3,3)
plt.ylim(0,3)
plt.xlim(0,3)

# 클래스별 출력 색상과 레이블
colors = ["magenta", "blue"]
class_id = ['class_0', 'class_1']

# 산점도
for c, i, c_name in zip(colors, [0,1], class_id):
    plt.scatter(xx[yy == i, 0], xx[yy== i, 1],
                c = c, s=5, alpha=0.3,
                edgecolors='none')
    plt.scatter(x[y==i,0], x[y==i,1],
                c=c,s=20, label=c_name)

plt.legend(loc='upper right')
plt.show()


