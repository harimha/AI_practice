from tensorflow import keras
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# 3개 출력 로지스틱 회귀
x = np.array([[1,1], [2,1], [1,2],
              [3,3], [3,2], [4,2],
              [1,3], [2,4], [3,4]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

denselayer = keras.layers.Dense(3, input_shape=(2,),
                                activation="softmax")
model = keras.Sequential([denselayer])
model.compile(optimizer=keras.optimizers.SGD(0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 400회씩 100회 반복 학습 = 40,000번
loss_hist =[]
steps = 400
for i in tqdm(range(100)):    
    model.fit(x, y, verbose=0, epochs=steps)    
    t_loss, t_acc = model.evaluate(x,y, verbose=0)
    loss_hist.append(t_loss)

print("\nTrain loss = ", t_loss)
print("Accuracy    = ", t_acc)

# 그래프 출력
loss_indices = range(0, len(loss_hist)*steps, steps)
plt.plot(loss_indices, loss_hist, 'k-',
         label="train set loss")
plt.title('Loss(Multinomial Logistic Regression)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# 학습 결과 분할된 영역
points = np.empty([2500, 2], np.float32)
k = 0
for i in range(50):
    for j in range(50):
        points[k][0] = i /10.0
        points[k][1] = j / 10.0
        k += 1
        
yy = model.predict(points)
cl = []
for i in range(yy.shape[0]):
    cl.append(np.argmax(yy[i]))

plt.ylim(0,5)
plt.xlim(0,5)
plt.scatter(points[:, 0], points[:, 1],
            c=cl, s=10, alpha=0.5)
plt.scatter(x[:,0], x[:, 1], c=y, s=40)
plt.show()



