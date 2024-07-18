import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the wine dataset
wine = load_wine()
X = pd.DataFrame(data=wine.data, columns=wine.feature_names)
y = wine.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a StandardScaler object
scaler = StandardScaler()

# Fit the scaler to the training data
scaler.fit(X_train)

# Transform the training and testing data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a sequential model
model = Sequential()

# Add a dense layer with 128 units and ReLU activation
model.add(Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)))

# Add another dense layer with 64 units and ReLU activation
model.add(Dense(64, activation='relu'))

# Add the output layer with 3 units and softmax activation
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_data=(X_test_scaled, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)
# save the model
model.save('wine_model.h5')
print('Model saved to wine_model.h5')
# load the model
from tensorflow.keras.models import load_model
model = load_model('wine_model.h5')
print('Model loaded from wine_ model.h5')
# make predictions
predictions = model.predict(X_test_scaled)
print('Predictions:', predictions) 