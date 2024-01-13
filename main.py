import tensorflow as tf
import logging
import numpy as np
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import matplotlib.pyplot as plt


#Setup the input in celcius and output in fahrenheit
celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

#Create dense model  
model = tf.keras.Sequential([
  tf.keras.layers.Dense(units=1, input_shape=[1])
])

#Compile dense model
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))


#Train the model
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")

plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
#plt.show()

print(model.predict([100.0]))

print("These are the layer variables: {}".format(model.get_weights()))