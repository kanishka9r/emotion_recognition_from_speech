# Step 1: Import necessary libraries
import os
import librosa
import numpy as np
np.complex = complex # For compatibility with librosa
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Bidirectional

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# This script loads audio files from the RAVDESS and TESS dataset, extracts MFCC features 

# Step 2: Define emotion code to label mapping
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}
# For TESS (folder names)
emotion_map_tess = {
    'angry': 'angry',
    'disgust': 'disgust',
    'fear': 'fearful',
    'happy': 'happy',
    'neutral': 'neutral',
    'ps': 'surprised',  # pleasant surprise
    'sad': 'sad'
}

# Step 3: Set path to the dataset folder
DATA_FOLDER = "ravdess" # Folder where all Actor folders are
DATA_FOLDER_TESS = "tess" # Folder where TESS emotion folders are
X = []  # Feature vectors (40 MFCC values per file)
y = []  # Emotion labels

# Step 5:Go through all Actor folders and files

#LOAD RAVDESS DATASET
for actor_folder in os.listdir(DATA_FOLDER):
    actor_path = os.path.join(DATA_FOLDER, actor_folder)
    if not os.path.isdir(actor_path): # Check if it’s really a folder (not a file)
        continue
    for file in os.listdir(actor_path): # Go through all files in the actor folder
        if file.endswith(".wav"):
            file_path = os.path.join(actor_path, file)
            emotion_code = file.split("-")[2]  # Get emotion from file name
            emotion_label = emotion_map.get(emotion_code)
            if emotion_label is None:
                continue # Skip unknown labels
            signal, sr = librosa.load(file_path, sr=None)  # Load the audio file
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)  # Extract 40 MFCCs (without averaging)
            mfcc = mfcc.T  # Transpose to shape [timesteps, features]
            X.append(mfcc) # Save features and label
            y.append(emotion_label)

# LOAD TESS DATASET
for emotion_folder in os.listdir(DATA_FOLDER_TESS):
    emotion_name = emotion_folder.lower() # Convert folder name to lowercase
    emotion_label = emotion_map_tess.get(emotion_name)
    if emotion_label is None:
        continue
    folder_path = os.path.join(DATA_FOLDER_TESS, emotion_folder)
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)
            signal, sr = librosa.load(file_path, sr=None)
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40).T
            X.append(mfcc)
            y.append(emotion_label)

#Step 6 : Print how many total audio samples were processed
print(f"Total samples collected: {len(X)}")

#Step 7 : Pad sequences to ensure uniform input shape
max_len = 350  #max no of frame 
# Pad all MFCCs to shape (max_len, 40)
X_padded = pad_sequences(X, maxlen=max_len, dtype='float32', padding='post', truncating='post')
# Now shape of X_padded is (num_samples, max_len, 40)
print("X_padded shape:", X_padded.shape)
# Flatten all MFCCs over samples and time (axis 0 and 1)
train_flat = X_padded.reshape(-1, X_padded.shape[2])  # Shape: (num_samples * timesteps, 40)
# Global mean and std across all MFCC features
mfcc_mean = np.mean(train_flat, axis=0)
mfcc_std = np.std(train_flat, axis=0)

# Step 8: Encode labels to integers and then to one-hot vectors
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded)

# Step 9: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_padded, y_onehot, test_size=0.2, stratify=y_encoded, random_state=42)
# Shape of input: (timesteps, features)
input_shape = (X_train.shape[1], X_train.shape[2])  #max frame, 40 MFCCs
# Normalize training
X_train_norm = (X_train - mfcc_mean) / mfcc_std
# Normalize validation (IMPORTANT: use same training mean and std!)
X_val_norm = (X_test - mfcc_mean) / mfcc_std

# Step 10: Build the LSTM model
model = Sequential()
# Add Conv1D layer: 32 filters, kernel size 3, ReLU activation
model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
model.add(MaxPooling1D(pool_size=2)) # Add MaxPooling1D layer: pool size 2
# Add another Conv1D layer: 64 filters, kernel size 3, ReLU activation
model.add(Conv1D(64, kernel_size=3, activation='relu'))  #
model.add(MaxPooling1D(pool_size=2))  # Add MaxPooling1D layer: pool size 2                  
# LSTM layer: 64 units, returns final hidden state
model.add(Bidirectional(LSTM(64 ,kernel_regularizer=l2(0.001))))
# Dropout: helps prevent overfitting
model.add(Dropout(0.3))
# Dense output layer (softmax for classification)
model.add(Dense(y_train.shape[1], activation='softmax'))  # Number of classes
# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Show summary
model.summary()

# Early stopping to prevent overfitting
early_stop = EarlyStopping(
    monitor='val_loss',         # What to watch
    patience=5,                 # Wait 5 epochs before stopping
    restore_best_weights=True  # Bring back the best model weights
)

# Step 11: Train the model
history = model.fit(
    X_train_norm , y_train,
    epochs=35,
    batch_size=32,
    validation_data=(X_val_norm , y_test) ,
    callbacks=[early_stop]
)

# Step 12: Evaluate the model
y_pred_probs = model.predict(X_val_norm)
y_pred = np.argmax(y_pred_probs, axis=1)  # Convert from probabilities to class indices
y_true = np.argmax(y_test, axis=1)        # Convert one-hot true labels to class indices
val_loss, val_accuracy = model.evaluate(X_val_norm , y_test)# Evaluate model on test/validation set
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")
print(classification_report(y_true, y_pred)) # Show precision, recall, F1-score for each emotio

# Step 13: Make confusion matrix
cm = confusion_matrix(y_true , y_pred) # Create confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues) # Display it
plt.title("Confusion Matrix - Emotion Recognition")
plt.show()

# Step 14: Save the model
model.save("emotion_lstm_model.h5")
print("emotion_lstm_model.h5 saved successfully.")

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.show()
