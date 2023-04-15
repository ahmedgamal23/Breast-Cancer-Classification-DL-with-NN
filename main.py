# Breast Cancer Classification DL with NN

# about data
#   - Diagnosis (M = malignant --> 0, B = benign --> 1)
#


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras


# data collection and preprocessing
data = pd.read_csv('data.csv')
data = data.iloc[: , 1:-1]

print(data.head())
print("===========================================")
print(data.shape)
print("===========================================")
print(data.info())
print("===========================================")
print(data.describe())
print("===========================================")
print(data.columns)

# Encoding target column (diagnosis)
data['diagnosis'] = np.where(data['diagnosis'].str.contains('M') , 0 , 1)
print("after encoding : \n" , data)

print(data.isnull().sum())
print("Number of values in each categories (Diagnosis (0 -- > M = malignant, 1 -- > B = benign) ) :\n " , data['diagnosis'].value_counts())
print(data.groupby('diagnosis').mean())

# Split data to X for features and Y for labels
X = data.iloc[: , 1:]
y = data.iloc[: , 0]

print(X.shape)
print("***********************************")
print(y.shape)

# Splitting the data into training data & Testing data
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=2)

print("for X : " , X.shape, X_train.shape, X_test.shape)
print("for y : " , y.shape, y_train.shape, y_test.shape)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# build Neural Network
'''
    input       hidden_layer    output
      x0=1          a1
      x1            a2            a1
      x2            a3            a2
      x3            a4
'''

tf.random.set_seed(123)

# build architecture for model
model = keras.Sequential([
            keras.layers.Flatten(input_shape=(30,)),            # input layer
            keras.layers.Dense(20 , activation='relu'),         # first hidden layer
            keras.layers.Dense(2 , activation='sigmoid')        # output layer
        ])

# compile model
model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy' , metrics=['accuracy'])

# fit x_train and y_train
fit_data = model.fit(X_train_std , y_train , epochs=10 , validation_split=0.1)

# Accuracy of the model on test data
loss , accuracy = model.evaluate(X_test_std , y_test)
print(accuracy)

# predict y hat
y_pred = model.predict(X_train_std) # PREDICT probability for each class (M or B)
print(y_pred.shape)
print(y_pred[0])
print("***********************")
print( "X_test_std : \n" , X_test_std)
print("***********************")
print(y_pred)

# converting the prediction probability to class labels
# argmax : return index for max value
y_pred_label = [np.argmax(i) for i in y_pred]
print(y_pred_label)


# visualize data Visualizing accuracy and loss
figure, axis = plt.subplots(1,1)
plt.plot(fit_data.history['accuracy'])
plt.plot(fit_data.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training data', 'validation data'], loc = 'lower right')
plt.show()

plt.plot(fit_data.history['loss'])
plt.plot(fit_data.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training data', 'validation data'], loc = 'upper right')
plt.show()

print("********************************************************************************************")
print("********************************************************************************************")
print("***********************     Building the predictive system    ******************************")
print("********************************************************************************************")
print("********************************************************************************************")

input_data = (11.76,21.6,74.72,427.9,0.08637,0.04966,0.01657,0.01115,0.1495,0.05888,0.4062,1.21,2.635,28.47,0.005857,0.009758,0.01168,0.007445,0.02406,0.001769,12.98,25.72,82.98,516.5,0.1085,0.08615,0.05523,0.03715,0.2433,0.06563)
# change the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one data point
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardizing the input data
input_data_std = scaler.transform(input_data_reshaped)

prediction = model.predict(input_data_std)
print(prediction)

prediction_label = [np.argmax(prediction)]
print(prediction_label)

if(prediction_label[0] == 0):
  print('The tumor is Malignant')

else:
  print('The tumor is Benign')



