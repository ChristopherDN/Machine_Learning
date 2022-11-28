import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from keras.preprocessing import image

trainingFile = '/Users/christophernielsen/PycharmProjects/Machine_Learning/Vegetable Images/train'
testFile = '/Users/christophernielsen/PycharmProjects/Machine_Learning/Vegetable Images/test'
validationFile = '/Users/christophernielsen/PycharmProjects/Machine_Learning/Vegetable Images/validation'



# image augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # change pixel from 0-255 to 0.0 to 1.0
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
training_set = train_datagen.flow_from_directory(
    trainingFile,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1. / 255)  # no zoom or shear, since this is test data
test_set = test_datagen.flow_from_directory(
    testFile,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

val_datagen = ImageDataGenerator(rescale=1. / 255)  # no zoom or shear, since this is test data
val_set = val_datagen.flow_from_directory(
    validationFile,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

# convolution
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=[64, 64, 3]))

# max pooling
model.add(MaxPool2D(pool_size=2, strides=2))

# 2nd layer with convolution and max pooling
model.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
model.add(MaxPool2D(pool_size=2, strides=2))

# flatten
model.add(Flatten())

# add fully connected layer (just as with classification or regression before)
model.add(Dense(units=128, activation="relu"))
# cnn.add(Dense(64, activation="sigmoid"))
model.add(Dense(units=15, activation="softmax"))#'sigmoid'

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=training_set, validation_data=val_set, epochs=3)

model.evaluate(test_set)

model.save("vegmodel.h5")

# singlePred = validationFile + '1197'
singlePred = '/Users/christophernielsen/PycharmProjects/Machine_Learning/Vegetable Images/validation/Bean/0023.jpg'#Bean

#singlePred = '/Users/christophernielsen/PycharmProjects/Machine_Learning/Vegetable Images/validation/Radish/1210.jpg'#Radish

test_image = image.image_utils.load_img(singlePred, target_size=(64, 64))
test_image = image.image_utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)  # add one extra dimension to hold batch column

result = model.predict(test_image / 255.0)
print(training_set.class_indices)

print("result is: ",(np.argmax(result)))

if np.argmax(result) == 0:
    print("its a Bean")  # depending on the value of training_set.class_indices
elif np.argmax(result) == 1:
        print("its a Bitter gourd")
elif np.argmax(result) == 2:
        print("its a Bottle gourd")
elif np.argmax(result) == 3:
        print("its a Brinjal")
elif np.argmax(result) == 4:
        print("its a Broccoli")
elif np.argmax(result)== 5:
        print("its a Cabbage")
elif np.argmax(result) == 6:
        print("its a Capsicum")
elif np.argmax(result) == 7:
    print("its a Carrot")
elif np.argmax(result) == 8:
        print("its a Cauliflower")
elif np.argmax(result) == 9:
        print("its a Cucumber")
elif np.argmax(result) == 10:
        print("its a Papaya")
elif np.argmax(result) == 11:
        print("its a Potato")
elif np.argmax(result) == 12:
        print("its a Pumpkin")
elif np.argmax(result) == 13:
        print("its a Radish")
elif np.argmax(result) == 14:
        print("its a Tomato")



