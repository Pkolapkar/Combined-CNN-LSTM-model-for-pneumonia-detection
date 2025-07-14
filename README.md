# Pneumonia Detection using Combined CNN-LSTM Model

This project presents a deep learning model that combines **Convolutional Neural Networks (CNN)** and **Long Short-Term Memory (LSTM)** to detect pneumonia from chest X-ray images. The integration of CNN for feature extraction and LSTM for sequence learning enhances classification accuracy.

## 📌 Objective

To develop a hybrid deep learning model that accurately detects pneumonia by analyzing chest X-ray images, providing a potential aid for radiologists and healthcare professionals.

---

## 🧠 Model Architecture

- **CNN Layer:** Extracts spatial features from input chest X-ray images.
- **LSTM Layer:** Captures temporal dependencies from CNN output.
- **Dense Layer:** Final classification layer for binary output (Pneumonia / Normal).

---

## 🗃️ Dataset

- **Source:** Kaggle Chest X-ray dataset
- **Categories:** 
  - Normal
  - Pneumonia
- **Link:** [Chest X-Ray Images (Pneumonia) | Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

---

## 🛠️ Tools & Technologies

- Python
- TensorFlow / Keras
- OpenCV
- NumPy, Pandas, Matplotlib
- Jupyter Notebook / Google Colab

---

## 📊 Results

- **Accuracy:** Achieved high classification accuracy on validation set
- **Performance Metrics:** Accuracy, Precision, Recall, F1-score
- Model performance evaluated through confusion matrix and ROC curves

---

## 🖼️ Poster

A scientific poster summarizing the model, dataset, methodology, and results is included in the repo:  
📄 [`3 Poster Template.docx`](./3%20Poster%20Template.docx)

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Pkolapkar/Combined-CNN-LSTM-model-for-pneumonia-detection.git
   cd Combined-CNN-LSTM-model-for-pneumonia-detection

pip install -r requirements.txt
python pneumonia_cnn_lstm.py
# or use Jupyter/Colab notebook
