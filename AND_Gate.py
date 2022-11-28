import numpy as np
import h5py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, load_model


model = Sequential()
#model.add(Dense(2, input_dim=2,activation='relu')) # relu is returning 0 or 1
#model.add(Dense(1, activation='relu'))
model.add(Dense(2, input_dim=2,activation='sigmoid'))  # sigmoid is returning 0.01 or 0.9
model.add(Dense(1, activation='sigmoid'))
adam = optimizers.Adam(learning_rate=0.01)
model.compile(loss='mean_squared_error', optimizer=adam)

# AND gate
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[0],[0],[1]])

history = model.fit(x, y, epochs=2500, batch_size=2, verbose=1)

print("result =", model.predict([[0,0]]),"expected  = 0")
print()
print("result =", model.predict([[0,1]]),"expected  = 0")
print()
print("result =", model.predict([[1,0]]),"expected  = 0")
print()
print("result =", model.predict([[1,1]]),"expected  = 1")
model.save('AND_model.h5')  # creates a HDF5 file 'my_model.h5'
myModel = load_model('AND_model.h5') # myModel is ready for predicting right away!