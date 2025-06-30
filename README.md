# Speech Emotion Recognition using CNN + BiLSTM

This project classifies human emotions from speech using a hybrid deep learning model built on top of MFCC features extracted from audio datasets (RAVDESS and TESS).

---

## Features

- Uses **MFCC (Mel-Frequency Cepstral Coefficients)** for audio feature extraction
- Combined **CNN** and **BiLSTM** architecture for better temporal and spatial analysis
- Includes **early stopping** and **dropout** to prevent overfitting
- Supports **multi-dataset training** (RAVDESS and TESS)
- Evaluates model using **confusion matrix** and **classification report**
- Achieved ~80% validation accuracy

---

##  Dataset Used

- [RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)](https://zenodo.org/record/1188976)
- [TESS (Toronto Emotional Speech Set)](https://tspace.library.utoronto.ca/handle/1807/24487)

---
## Model Architecture

- `Conv1D` → `MaxPooling1D`
- `Conv1D` → `MaxPooling1D`
- `Bidirectional(LSTM)`
- `Dropout`
- `Dense(softmax)`

---

##  Future Enhancements

- Use **attention mechanism** on top of LSTM
- Add **augmentation** (noise, pitch shift, time stretch)
- Support for **real-time prediction** using microphone input
- Train on more datasets like **EMO-DB**, **SAVEE**

---

##  Tech Stack

- Python, TensorFlow, Keras
- Librosa for feature extraction
- Scikit-learn for evaluation

---

##  How to Run

1. Download RAVDESS and TESS datasets into `ravdess/` and `tess/` folders.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run training:
   ```bash
   python emotion_lstm_model.py
4. After training completes, the model will:
Display a confusion matrix
Show accuracy plots
Save the model as emotion_lstm_model.h5
