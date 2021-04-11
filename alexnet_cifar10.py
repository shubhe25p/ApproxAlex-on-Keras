
from tensorflow import keras
from keras.layers import *
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
from keras.utils import to_categorical


import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

X_train = x_train/255
X_test = x_test/255
Y_train = to_categorical(y_train)
Y_test = to_categorical(y_test)

def approx_alex(input_shape):

  X_input = Input(input_shape)
  #X=38,38,3
  X = ZeroPadding2D((3,3))(X_input)

  X = Conv2D(32,(7,7),strides=(1,1),name='conv0', kernel_initializer=glorot_uniform(seed=0))(X)
  #X=32,32,32
  X = BatchNormalization(axis=3,name='bn0')(X)

  X = Activation('relu')(X)
  #31,31,32
  X = MaxPooling2D((2,2),strides=(2,2),name='max-pool0')(X)

  X = Conv2D(64,(5,5),strides=(1,1),name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)

  X = BatchNormalization(axis=3, name='bn1')(X)

  X = Activation('relu')(X)

  X = MaxPooling2D((2,2), strides=(1,1),name='max-pool1')(X)

  X = Conv2D(96,(3,3),strides=(1,1),name='conv2', kernel_initializer=glorot_uniform(seed=0))(X)

  X = BatchNormalization(axis=3, name='bn2')(X)

  X = Activation('relu')(X)
  
  X = Conv2D(128,(3,3),strides=(1,1),name='conv3', kernel_initializer=glorot_uniform(seed=0))(X)

  X = BatchNormalization(axis=3, name='bn3')(X)

  X = Activation('relu')(X)

  X = Conv2D(256,(3,3),strides=(1,1),name='conv4', kernel_initializer=glorot_uniform(seed=0))(X)

  X = BatchNormalization(axis=3, name='bn4')(X)

  X = Activation('relu')(X)

  X = MaxPooling2D((2,2), strides=(1,1),name='max-pool2')(X)

  X = Flatten()(X)

  X = Dense(2048, activation='relu', name='fc1', kernel_initializer=glorot_uniform(seed=0))(X)
  X = Dense(2048, activation='relu', name='fc2', kernel_initializer=glorot_uniform(seed=0))(X)
  X = Dense(10, activation='softmax', name='fc3', kernel_initializer=glorot_uniform(seed=0))(X)

  model=Model(inputs = X_input, outputs = X, name='approx_alex')

  return model

model = approx_alex(X_train.shape[1:])

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#model.fit(X_train, Y_train, epochs=32, batch_size=64)

#history=model.fit(X_train, Y_train, epochs=30, batch_size=32)

h2=model.fit(X_train, Y_train, epochs=40, batch_size=32)

plt.plot(h2.history['accuracy'])
plt.title('Model Acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.show()

#import kerastuner as kt
#
#tuner = kt.Hyperband(
#    model,
#    objective='accuracy',
#    max_epochs=30,
#    hyperband_iterations=2)

preds=model.evaluate(X_test, Y_test)
print()
print("Loss="+ str(preds[0]))
print("Test Accuracy=" + str(preds[1]))

