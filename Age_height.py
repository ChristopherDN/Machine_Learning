import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers


model = Sequential()
model.add(Dense(8, input_dim=1, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(6, activation='relu'))
#model.add(Dense(units=1, activation='linear')) eller 2 gange denne
model.add(Dense(units=1, activation='linear'))
adam = optimizers.Adam(learning_rate=0.01)
model.compile(loss='mean_squared_error', optimizer=adam)



x = np.array([[1],
              [3],
              [5],
              [7],
              [9],
              [11],
              [13],
              [15],
              [17],
              [19],
              [21],
              [23],
              [25],
              [27],
              [29],
              [40]])

y = np.array([[76],
              [96],
              [112],
              [126],
              [136],
              [148],
              [160],
              [174],
              [178],
              [182],
              [182],
              [182],
              [183],
              [183],
              [184],
              [184]])

history = model.fit(x, y, epochs=2500, batch_size=2, verbose=1)
print('Enter your age to determine your height:')
value = int(input())
model.save('Age_height.h5')  # creates a HDF5 file 'my_model.h5'
myModel = load_model('Age_height.h5') # myModel is ready for predicting right away!
prediction = model.predict([value])
print(prediction)
