import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

# Alternative 1: Use your own Github account, if you have images there
# Alternative 2: Download these images and upload the folder to Colab
targetSize = 64  # pixel dimension after ImageDataGenerator has processed
no_of_filters = 7  # how many different filters
color = 'rgb'  # use "rgb" for color images
classMode = 'categorical'  # use 'categorical' for >2 class, 'binary' for two-class problems
trainingFiles = '/Users/christophernielsen/PycharmProjects/Machine_Learning/Vegetable Images/train'  # change according to your setup

train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # change pixel value from 0-255 to 0.0 - 1.0
    shear_range=0.2,  # distort the image sideways
    zoom_range=0.2,
    horizontal_flip=True
)
# you can also use shear_range, zoom_range and horizontal_flip to "disturb" the images. This will # make the model more robust, and will
# reduce overfitting.

# Optional:
test_datagen = ImageDataGenerator(rescale=1. / 255)  # no zoom or shear, since this is test data
training_set = train_datagen.flow_from_directory(
    trainingFiles,  # path to folder with images
    target_size=(targetSize, targetSize),  # size of output image, f.x. 28 x 28 pixel
    batch_size=32,  # how many images to load at a time
    class_mode=classMode,  # use 'categorical' for >2 class, 'binary' for two-class problems
    color_mode=color)  # use 'grayscale' for black/white, 'rgb' for color

# Optional:

test_set = test_datagen.flow_from_directory(
    '/Users/christophernielsen/PycharmProjects/Machine_Learning/Vegetable Images/test',
    # path to folder with test images
    target_size=(targetSize, targetSize),  # size of output image, f.x. 28 x 28 pixel
    batch_size=32,  # how many images to load at a time
    class_mode='categorical',  # use 'categorical' for >2 class, 'binary' for two-class problems
    color_mode=color)  # use 'grayscale' for black/white, 'rgb' for color

model = Sequential()  # instantiate new model object.

model.add(Conv2D(filters=32,  # specify number of filters. Higher number for more complex images.
                 kernel_size=3,  # size of filter - typically 3, as in 3x3
                 activation="relu",  # activation function, often 'relu' on layers before last
                 input_shape=[targetSize, targetSize, 3]))  # dimension of image, coming in from training set.
# here 28x28 pixel. '1' is number of color channels. B/W = 1, Color = 3.


model.add(MaxPool2D(pool_size=2, strides=2))  # reduce the image size. Here 4 pixels will become 1 pixel.
# pool_size is the size of the square which will be converted to just one pixel. Often 2, as in 2x2.
# strides is how many pixels to move to the right after each pooling operation. Often 2.


# Optional: You may add further stack(s) of Conv2D + MaxPool2D if needed, using model.add(...)


model.add(Flatten())  # Convert the output layer to a single column, an array of shape (length, 1).
# add fully connected layer (just as with DNN Step 11 above)
model.add(Dense(units=64, activation="relu"))  # here 4 output neurons for a simple problem

# model.add(Dense(units=15, activation="sigmoid"))  # one single output neuron and sigmoid ac. func.
model.add(Dense(units=15, activation="softmax"))  # one single output neuron and sigmoid ac. func.

adam = Adam(learning_rate=0.01)  # use the Adam optimizer, and specify learning-rate
# model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# here parameters are set to solve a two-class problem. Hence binary_crossentropy.
# metrics = ['accuracy'] should be used for classification (not regression)


model.fit(x=training_set, epochs=5)  # This will train the model
# if you made a test_set in step 5, then provide it as parameter like this: validation_data=test_set
# The model will evaluate against the test set at the end of each epoch. The model will not be trained on these images.

print(training_set.class_indices)  # get the indices for each class. F.x. horizontal = 0, vertical = 1
# Optional:
model.evaluate(
    test_set)  # If you made a test_set in step 5, this will predict ALL the images and return the accuracy in %.
# the reason it can be evaluated is, that the labels are automatically inferred from the folder structure of the image files.
singlePred = '/Users/christophernielsen/PycharmProjects/Machine_Learning/Vegetable Images/validation/Potato/1204.jpg'
test_image = image.image_utils.load_img(singlePred, target_size=[targetSize, targetSize], color_mode=color)
# here set the SAME parameters as on the training in Step 5.

test_image = image.image_utils.img_to_array(test_image)  # convert image to array
test_image = np.expand_dims(test_image, axis=0)  # add one extra dimension to hold batch.
# axis=0 means that a new dimension will be added, such that test_image.shape goes from (28, 28, 1) to (1, 28, 28, 1). This is required by Tensorflow.

result = model.predict(test_image / 255.0)  # remember to divide each pixel value by 255.0
print(np.argmax(result))

# print("result is: " + str(result[0][0]))
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
elif np.argmax(result) == 5:
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

"""
import cv2
vid = cv2.VideoCapture(0)

while (True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # If needed, convert the frame to grayscale
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.putText(frame, 'It is a potato', (10,100),
                cv2.FONT_HERSHEY_SIMPLEX,
                4,(255,255,255), 4, 2)

    # Display the resulting frame
    cv2.imshow('Camera feed', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
"""
