#%%
# import Libraries
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
import numpy as np

#%% load data

train_data = pd.read_csv("mnist_train.csv")
test_data = pd.read_csv("mnist_test.csv")

print(train_data.head())
print(test_data.head())

#%%

train_data = train_data.values
test_data = test_data.values

print("train data: ", train_data)
print("test data: ", test_data)
#%% train dataset
np.random.shuffle(train_data)
x_train = train_data[:,1:].reshape(-1,28,28,1)/255.0
print(x_train.shape)
y_train = train_data[:,0].astype(np.int32)
y_train = to_categorical(y_train, num_classes=len(set(y_train)))

#%% test dataset

x_test = test_data[:,1:].reshape(-1,28,28,1)/255.0
print(x_test.shape)
y_test = test_data[:,0].astype(np.int32)
y_test = to_categorical(y_test, num_classes=len(set(y_test)))


#%% one image show
import warnings
warnings.filterwarnings("ignore")

img = x_train.reshape(60000,28,28)
plt.imshow(img[100,:,:])
plt.legend()
plt.axis("off")
plt.show()


#%%
number_of_class = y_train.shape[1]

#%% cnn model

model = Sequential()

model.add(Conv2D(input_shape = (28,28,1), filters = 32, kernel_size = (3,3),padding='same'))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size = (2, 2)))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size = (2, 2)))


model.add(Conv2D(filters = 128, kernel_size = (3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))


model.add(Conv2D(filters = 128, kernel_size = (3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(units=256))
model.add(Activation('tanh'))
model.add(Dropout(rate=0.3))
model.add(Dense(units=number_of_class))
model.add(Activation('softmax'))

#%% run model

model.compile(loss='categorical_crossentropy',
              optimizer='adam', 
              metrics=['accuracy'])
hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=11, batch_size=1024)

#%% save model
model.save_weights('cnn_with_mnist.h5')
#%% save history

import json
with open('cnn_mnist.json','w') as f:
    json.dump(hist.history, f)
    
#%% load history

import codecs
with codecs.open('cnn_with_mnist.json','r') as f:
    h = json.loads(f.read())


#%% visulazition accuracy - loss

print(hist.history.keys()) # val_loss, val_acc, loss, acc
# Loss graph
plt.plot(hist.history['loss'], label = "Train Loss")
plt.plot(hist.history['val_loss'], label = "Validation Loss")
plt.legend()
plt.show()


# Acc graph
plt.figure()
plt.plot(hist.history['acc'], label = "Train Accuracy")
plt.plot(hist.history['val_acc'], label = "Test Accuracy")
plt.legend()
plt.show()



#%%
model.summary()

