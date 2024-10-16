import pandas as pd
from keras import Sequential
from keras._tf_keras.keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('/home/ritam/Documents/LocalCodes/machine-learning/student-mat.csv')

# Display the first few rows
print(data.head())

# Select relevant features and target variable
# Here we choose a subset of columns that could be relevant for prediction
# You might need to adjust the columns based on exploratory data analysis

# features er ei gulor value dataset a already ache, egulo theke predict kore g3(grade3) er value ta predict korche,
# je g3 te ei 5 ta value ki ki hobe
features = ['studytime', 'failures', 'absences', 'G1', 'G2']
# features er ei gulor value dataset a already ache, egulo theke predict kore g3(grade3) er value ta predict korche,
# je g3 te ei 5 ta value ki ki hobe

target = 'G3'

X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the neural network model
model = Sequential()
model.add(Dense(64, activation='leaky_relu', input_dim=5))  # Input layer with 64 neurons
model.add(Dropout(0.2))  # Dropout layer to prevent overfitting
model.add(Dense(128, activation='leaky_relu'))  # Hidden layer with 128 neurons
model.add(Dropout(0.2))
model.add(Dense(256, activation='leaky_relu'))  # Hidden layer with 256 neurons
model.add(Dropout(0.2))
model.add(Dense(128, activation='leaky_relu'))  # Hidden layer with 128 neurons
model.add(Dropout(0.2))
model.add(Dense(64, activation='leaky_relu'))  # Hidden layer with 64 neurons
model.add(Dropout(0.2))
model.add(Dense(32, activation='leaky_relu'))  # Hidden layer with 32 neurons
model.add(Dense(1))  # Output layer with 1 neuron for regression output

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train the model
model.fit(X_train, y_train, epochs=500, batch_size=10, validation_split=0.2)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f'Mean Absolute Error on Test Set: {mae}')

# Make predictions
predictions = model.predict(X_test)

# Display the first few predictions
print(predictions[:5])

model.save('stu_model.keras')
print("file saved successfully")
