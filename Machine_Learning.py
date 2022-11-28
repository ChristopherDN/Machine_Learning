import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers


model = Sequential()
#model.add(Dense(6, input_dim=6, activation='relu'))
#model.add(Dense(3, activation='relu'))
model.add(Dense(6, input_dim=6, activation='sigmoid'))
model.add(Dense(3, activation='sigmoid'))
adam = optimizers.Adam(learning_rate=0.01)
model.compile(loss='mean_squared_error', optimizer=adam)

x = np.array([[0.27, 0.24,  1, 0,1,0],
              [0.48, 0.98, -1, 1,0,0],
              [0.33, 0.44, -1, 0,0,1],
              [0.30, 0.29,  1, 1,0,0],
              [0.66, 0.65, -1, 0,1,0]])

"""y = np.array([[0.43, 0.37, 0.20],
             [0.37, 0.20, 0.43],
             [0.37, 0.20, 0.43],
             [0.20, 0.43, 0.37],
             [0.43, 0.37, 0.20]])"""

y = np.array([[1, 0, 1],
              [0, 1, 1],
              [0, 1, 1],
              [1, 1, 0],
              [1, 0, 1]])


history = model.fit(x, y, epochs=2500, batch_size=2, verbose=1)

#prediction = model.predict([[0.38, 0.51, -1, 0,1,0]])
#print(prediction)
#print("Result =", model.predict([[0.38, 0.51, -1, 0,1,0]]))
#print("0.37  0.43  0.20 = Republican\n0.20  0.37  0.43 = Independent\n0.43  0.37  0.20 = Democrat")
print("Result =", model.predict([[0.38, 0.51, -1, 0,1,0]]))
print("0  1  1 = Republican\n1  1  0 = Independent\n1  0  1 = Democrat")