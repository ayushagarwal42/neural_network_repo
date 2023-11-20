import tensorflow as tf
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

# Generate some sample data
# Assuming input sequences of length 10 with 5 features
# X_train shape: (batch_size, sequence_length, input_features)
# y_train shape: (batch_size, output_features)
X_train = tf.random.normal((1000, 10, 5))
y_train = tf.random.normal((1000, 1))

# Build the RNN model
model = Sequential()
model.add(SimpleRNN(units=32, input_shape=(10, 5), activation='relu'))
model.add(Dense(units=1))  # Output layer with one neuron for regression

# Compile the model
model.compile(optimizer='adam', loss=tf.keras.losses.mean_squared_error, metrics=['mae'])

# Display the model summary
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
