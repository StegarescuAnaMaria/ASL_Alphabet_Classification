import numpy as np
import os
from skimage.transform import resize
from skimage.io import imread
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import validation_curve
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.feature import hog


#Function for plotting train vs validation graphics
def plot_figure(Y_train, Y_val, X, title=None, legend=None, labels=None):
  plt.figure()
  if title:
    plt.title(title)
  line1, = plt.plot(X, Y_train, 'bo--')
  line2, = plt.plot(X, Y_val, 'ro--')
  if legend:
    line1.set_label(legend[0])
    line2.set_label(legend[1])
    plt.legend()
  if labels:
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
  plt.show()



def feature_extraction(list_, sub_folder, folder_hog_img):
  for i, image_name in enumerate(list_):
    image = imread(folder + "/" + sub_folder + "/" + image_name)
    labels[i] = 0
    resized_arr = resize(image, (64, 64))
    
    fd, hog_img = hog(resized_arr, orientations=9, pixels_per_cell=(2, 2),
              cells_per_block=(1, 1), visualize=True, channel_axis=2)
    plt.imsave(folder_hog_img + "/hog_" + image_name, hog_img, cmap='gray')
    
    
    
def build_cnn_model_complex():
    model_cnn = models.Sequential()
    model_cnn.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
    model_cnn.add(layers.MaxPooling2D((2, 2)))
    model_cnn.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model_cnn.add(layers.MaxPooling2D((2, 2)))
    model_cnn.add(layers.Conv2D(16, (3, 3), activation='relu'))

    model_cnn.add(layers.Flatten())
    model_cnn.add(layers.Dense(29, activation="softmax"))

    model_cnn.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model_cnn
    


def build_cnn_model_simple():
    model_cnn = models.Sequential()
    model_cnn.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
    model_cnn.add(layers.MaxPooling2D((2, 2)))
    model_cnn.add(layers.Conv2D(16, (3, 3), activation='relu'))

    model_cnn.add(layers.Flatten())
    model_cnn.add(layers.Dense(29, activation="softmax"))

    model_cnn.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model_cnn


def build_rnn_model():
    model_rnn = models.Sequential()
    model_rnn.add(layers.LSTM(units=32, activation='relu'))
    model_rnn.add(layers.Flatten())
    model_rnn.add(layers.Dense(29, activation="softmax"))
    return model_rnn


def transform_to_keras_class(model_name):
    if model_name=='cnn_s':
        return KerasClassifier(build_cnn_model_simple(), verbose=0)
    elif model_name == 'cnn':
        return KerasClassifier(build_cnn_model_complex(), verbose=0)
    else:
        return KerasClassifier(build_rnn_model(), verbose=0)
    
    
    
def tune_model(model, param_name, param, val_data, val_labels, train_data, train_labels):
    history = model.fit(img_train, labels_train, epochs=30, batch_size=15,
                    validation_data=(img_val, labels_val))
    val_loss, val_acc = model.evaluate(val_data, val_labels, verbose=2)
    train_loss, train_acc = model.evaluate(train_data, train_labels, verbose=2)
    
    return train_acc, val_acc, train_loss, val_loss








    
    
folder = "./dataset_asl/asl_alphabet_train"

# list equal to ['A', 'B', 'C'....]
sub_folders = os.listdir(folder)    

folder_hog_img = "./hog_images_64"


#Block of code for writing the resized hog images to ./hog_images_64 path
"""
for sub in sub_folders:
    img_names = os.listdir(folder + "/" + sub)
    feature_extraction(img_names, sub, folder_hog_img)
"""

#Reading the hog data
img_names = os.listdir(img_hog_folder)

images_hog_gray = np.zeros((len(img_names), 64, 64))
for i in range(len(img_names)):
    images_hog_gray[i] = imread(img_hog_folder + "/" + img_names[i], as_gray=True)
    
    
labels = np.zeros((len(img_names),))
for i in range(29):
    labels[i*3000 : (i+1)*3000] = i
    
#Train/Validation/Test Split    
img_train, img_test_val, labels_train, labels_test_val = train_test_split(images_hog_gray, labels, test_size=0.4, random_state=42)
img_test, img_val, labels_test, labels_val = train_test_split(img_test_val, labels_test_val, test_size=0.3, random_state=75)


#Some models need 4 dimensional data
img_train_rgb = np.reshape(img_train, (-1, 64, 64, 1))
img_test_rgb = np.reshape(img_test, (-1, 64, 64, 1))
img_val_rgb = np.reshape(img_val, (-1, 64, 64, 1))



#CNN test
model_cnn = models.Sequential()
model_cnn.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
model_cnn.add(layers.MaxPooling2D((2, 2)))
model_cnn.add(layers.Conv2D(16, (3, 3), activation='relu'))
model_cnn.add(layers.MaxPooling2D((2, 2)))
model_cnn.add(layers.Conv2D(16, (3, 3), activation='relu'))

model_cnn.add(layers.Flatten())
model_cnn.add(layers.Dense(29, activation="softmax"))

model_cnn.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(img_train_rgb, labels_train, epochs=10, batch_size=15,
                    validation_data=(img_val_rgb, labels_val))

test_loss, test_acc = model_cnn.evaluate(img_test_rgb, labels_test, verbose=2)

print(test_loss, test_acc)

plt.plot(history.history['accuracy'], label='Train accuracy')
plt.plot(history.history['val_accuracy'], label = 'Val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')



#CNN Validation

train_scores, val_scores = validation_curve(transform_to_keras_class('cnn'), img_train_rgb, labels_train, param_name="epochs", param_range=[5, 7, 10, 15], scoring="accuracy", n_jobs=2)
train_scores_mean = np.mean(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
plot_figure(train_scores_mean, val_scores_mean, [5, 7, 10, 15], title=None, legend=['Train', 'Validation'],
            labels=['Number of Epochs', 'Accuracy'])


train_scores, val_scores = validation_curve(transform_to_keras_class('cnn'), img_train_rgb, labels_train, param_name="batch_size", param_range=[10, 15, 25], scoring="accuracy", n_jobs=2)
train_scores_mean = np.mean(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
plot_figure(train_scores_mean, val_scores_mean, [10, 15, 25], title=None, legend=['Train', 'Validation'],
            labels=['Batch Size', 'Accuracy'])


train_scores, val_scores = validation_curve(transform_to_keras_class('cnn_s'), img_train_rgb, labels_train, param_name="epochs", param_range=[10, 15, 25, 30], scoring="accuracy", n_jobs=2)
train_scores_mean = np.mean(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
plot_figure(train_scores_mean, val_scores_mean, [10, 15, 25, 30], title=None, legend=['Train', 'Validation'],
            labels=['Number of Epochs', 'Accuracy'])



#RNN test
model_rnn = models.Sequential()
model_rnn.add(layers.LSTM(units=32, activation='relu'))
model_rnn.add(layers.Flatten())
model_rnn.add(layers.Dense(29, activation="softmax"))

model_rnn.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model_rnn.fit(img_train, labels_train, epochs=50, 
                    validation_data=(img_val, labels_val))

plt.plot(history.history['accuracy'], label='Train accuracy')
plt.plot(history.history['val_accuracy'], label = 'Val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model_rnn.evaluate(img_test, labels_test, verbose=2)

print(test_loss, test_acc)



#RNN Validation____________________________________________________________________________
train_scores, val_scores = validation_curve(transform_to_keras_class('rnn'), img_train, labels_train, param_name="batch_size", param_range=[10, 15, 25], scoring="accuracy", n_jobs=2)
train_scores_mean = np.mean(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
plot_figure(train_scores_mean, val_scores_mean, [10, 15, 25], title=None, legend=['Train', 'Validation'],
            labels=['Batch Size', 'Accuracy'])


train_scores, val_scores = validation_curve(transform_to_keras_class('rnn'), img_train, labels_train, param_name="epochs", param_range=[30, 60, 75], scoring="accuracy", n_jobs=2)
train_scores_mean = np.mean(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
plot_figure(train_scores_mean, val_scores_mean, [30, 60, 75], title=None, legend=['Train', 'Validation'],
            labels=['Number of Epochs', 'Accuracy'])
#________________________________________________________________________________________





#__________________________________________________________________________________________
#Finding out the best function for RNN

param_range = ['sigmoid', 'elu', 'tanh', 'relu']

train_acc, val_acc, train_loss, val_loss = [], [], [], []

for param in param_range:
    train_acc_, val_acc_, train_loss_, val_loss_ = tune_model(build_rnn_model(param), 'epochs', param, img_val, labels_val, img_train, labels_train)
    train_acc.append(train_acc_)
    val_acc.append(val_acc_)   
    train_loss.append(train_loss_)
    val_loss.append(val_loss_)
    
plot_figure(train_acc, val_acc, [1, 2, 3, 4], title=None, legend=['Train', 'Validation'],
            labels=['Identity, Logistic, Tanh, ReLU', 'Accuracy'])

plot_figure(train_loss, val_loss, [1, 2, 3, 4], title=None, legend=['Train', 'Validation'],
            labels=['Identity, Logistic, Tanh, ReLU', 'Loss'])
#__________________________________________________________________________________________




#Preparing the data for MLPClassifier
scalify = StandardScaler()

X_train_scl = scalify.fit_transform(img_train.reshape(-1, 64*64))
X_test_scl = scalify.fit_transform(img_test.reshape(-1, 64*64))
X_val_scl = scalify.fit_transform(img_val.reshape(-1, 64*64))


#Validating the MLPClassifier
train_scores, valid_scores = validation_curve(estimator=MLPClassifier(), X=X_train_scl, y=labels_train, scoring='accuracy', param_name='solver', param_range=['lbfgs', 'sgd', 'adam'], cv=5)
train_scores_mean = np.mean(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
plot_figure(train_scores_mean, val_scores_mean, [1, 2, 3], title="Solvers: LBFGS, SGD, Adam", legend=['Train', 'Validation'],
            labels=['LBFGS, SGD, Adam', 'Accuracy'])

train_scores, valid_scores = validation_curve(estimator=MLPClassifier(sollver='lbfgs'), X=X_train_scl, y=labels_train, scoring='accuracy', param_name='activation', param_range=['identity', 'logistic', 'tanh', 'relu'], cv=5)
train_scores_mean = np.mean(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
plot_figure(train_scores_mean, val_scores_mean, [1, 2, 3, 4], title="Activation function", legend=['Train', 'Validation'],
            labels=['Identity, Logistic, Tanh, ReLU', 'Accuracy'])







