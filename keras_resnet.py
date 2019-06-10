#!/usr/bin/env python
import numpy as np
import keras

from keras.preprocessing import image
from keras.applications import resnet50

#Load Keras ResNet50 pre-trained model
model = resnet50.ResNet50()

#Load the image file,resizing it to 224x224 pixels as required by resnet model
img = image.load_img('green_apple.jpg',target_size = (224,224))

#convert the image to numpy array
x = image.img_to_array(img)

# add a fourth dimension since expects a list of images
x = np.expand_dims(x,axis=0)

#scale the input image to the range used in trained network
x = resnet50.preprocess_input(x)

#Run the image through the model to make a prediction
predictions = model.predict(x)

#Look up the names of the predicted classes. By default, it gives the top 5 most closely matched results.
#Here,we are setting it to top 4
predicted_classes = resnet50.decode_predictions(predictions,top=4)

#gives the top 4 likelihood of the image
for imagenet_id,name,likelihood in predicted_classes[0]:
    with open('output.txt', 'a+') as f:
        f.write(print(" - {}: {:2f} likelihood".format(name,likelihood)))
        f.write('\n')
