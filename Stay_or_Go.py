import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score

dataFrame = pd.read_csv('/Users/christophernielsen/PycharmProjects/Machine_Learning/customer_staying_or_not.csv') # replace with your file
dataFrame.head()  # see the first 5 rows
dataFrame.isnull().sum()  # will count number of rows
dataFrame.dropna(inplace=True)  # will remove rows with # Thanks to David F. R. Petersen
pd.set_option('display.max_columns', None) # print all columns
X = dataFrame.iloc[:,3:13] # select relevant rows and columns to X (here for example all rows and columns 5,6,7,8,9,10 and 11)
y = dataFrame.iloc[:, -1] # select column(s) for y (here all rows and only the last column)
# Capital X and lower-case y comes from Linear Algebra. The input is often a 2D array (matrix, named X) while the output is often a 1D array (vector, named y)
X = pd.get_dummies(X) # convert ALL text-columns to categorical variables (One Hot encoding), e.g. gender, country etc.
columnNames = list(X.columns) # grab column-names before converting to numpy array
X = X.values # convert from Pandas dataframe to numpy array
y = y.values # convert from Pandas dataframe to numpy array

scaler = StandardScaler()
X = scaler.fit_transform(X) # calculate mean and standard deviation and convert dataframe to numpy array
# only use this, if the data is outside 0.0 … 1.0

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42 )

model = Sequential()
model.add(Dense(10,activation='relu')) # 4 outputs. It will automatically adapt to number inputs
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='sigmoid')) # Final output node for prediction. In this case, only one output neuron

adam = Adam(learning_rate=0.001) # you may have to change learning_rate, if the model does not learn.
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
# use loss = 'binary_crossentropy' for two-class classification.
# use loss = 'categorical_crossentropy' for multi-class classification.
## use loss = 'mse' (Mean Square Error) for regression (e.g. the Age,Height exercise).
## use metrics = ['accuracy']. It shows successful predictions / total predictions
#
model.fit(X_train,y_train,epochs=200, verbose=0)  # does the actual WORK !. verbose=1 will show output. 0 = no output.
loss = model.history.history['loss']
#sns.lineplot(X=range(len(loss)),y=loss)
model.evaluate(X_test,y_test,verbose=1)
#
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5) # creates a new array with true/false based on the boolean test
#
cm = confusion_matrix(y_test, y_pred)
print(cm)
#
## will return a 2D array like this (random numbers):  jart@kea.dk
## [[6432   326]
# # [ 481  1190]]
#
## interpretation:
## Top-left: 6432 correct predictions of 0.
# Top-right: -6 incorrect predictions of 1, when the y_test was 0.
# Bottom-left: 481 incorrect predictions of 0, when the y_test was 1.
# Bottom-right: 1190 correct predictions of 1
#print(dataFrame.iloc[0:int(len(dataFrame)*0.8),3:13])
#print(dataFrame.iloc[int(len(dataFrame)*0.8):(len(dataFrame)),3:13])
#to print the first 6 rows and the first 4 columns of a numpy 2d-array:

#print(columnNames) # first print column names, so you can enter new data in the correct columns
new_value = [[600, 40, 3, 60000, 2, 1, 1, 50000, 1, 0, 0, 0, 1]] # enter new data in 2D array. Only numbers + dummy variables.
new_value = scaler.transform(new_value) # Don't forget to scale!
model.predict(new_value)
print("Result = ",(model.predict(new_value)))
print( "0 = stay \n1 = leaves ")
model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
myModel = load_model('my_model.h5') # myModel is ready for predicting right away!

