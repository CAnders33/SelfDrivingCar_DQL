import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("training_data.csv")

df["State"] = df["State"].apply(lambda x: np.array(eval(x)))
df["Action"] = df["Action"].apply(lambda x: np.array(eval(x)))
df["Next State"] = df["Next State"].apply(lambda x: np.array(eval(x)))

X = np.array(df["State"].tolist())  
y = np.array(df["Action"].tolist())  

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Check for an existing model
model_path = "car_dql_model.h5"

if os.path.exists(model_path):
    print("🔄 Loading existing model for further training...")
    model = tf.keras.models.load_model(model_path)
else:
    print("🆕 No existing model found. Creating a new model...")
    model = Sequential([
        Dense(64, activation="relu", input_shape=(len(X_train[0]),)),  
        Dense(64, activation="relu"),
        Dense(3, activation="linear")  
    ])
    model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))

# Print model summary
model.summary()

# Train the model with more data
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)

# Save the updated model
model.save(model_path)
print("✅ Model updated and saved as 'car_dql_model.h5'")
