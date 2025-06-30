# 🎙️ Speech Emotion Recognition using CNN + BiLSTM

This project classifies emotions from speech using MFCC features and a hybrid CNN + BiLSTM model.

## 📂 Datasets Used
- [RAVDESS](https://zenodo.org/record/1188976)
- [TESS](https://tspace.library.utoronto.ca/handle/1807/24487)

## 🎯 Emotions Covered
- Angry, Disgust, Fearful, Happy, Neutral, Sad, Surprised

## 🧠 Model Architecture
- `Conv1D → MaxPooling → Conv1D → MaxPooling → BiLSTM → Dropout → Dense`
- Trained using categorical crossentropy + Adam optimizer

## 📊 Accuracy
- Achieved **~70% validation accuracy** on mixed RAVDESS + TESS dataset

## 📈 Evaluation
- Includes confusion matrix and classification report
- Visualizes training and validation accuracy across epochs

## 🛠️ Requirements
Install dependencies using:

```bash
pip install -r requirements.txt