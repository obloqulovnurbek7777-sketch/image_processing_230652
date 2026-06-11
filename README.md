# 🚦 German Traffic Sign Recognition Benchmark (GTSRB) using CNN

This repository contains a high-performance Deep Learning pipeline built with **TensorFlow 2.x** and **Keras** to classify traffic signs from the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The final model achieves an outstanding verification accuracy of **99.94%**.

---

## 📂 Project Structure
Following standard machine learning repository structures, the project is organized as follows:

```text
ProjectName/
├── README.md
├── requirements.txt
├── dataset/                 # Raw/Preprocessed image files (Train, Val, Test)
├── src/                     # Core python scripts (Preprocessing, Training)
├── notebooks/               # Google Colab / Jupyter Notebook files
├── results/                 # Training curves, confusion matrix, saved models
├── report/                  # Technical documentation and final project PDF
├── presentation/            # PowerPoint slides for project defense
└── documentation/           # API and usage guides

---
## 📊 Dataset Overview
The project utilizes the official **German Traffic Sign Recognition Benchmark (GTSRB)** dataset hosted on Kaggle. You can access and download the dataset directly from here:

🔗 **[Kaggle GTSRB Dataset Link](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)**

* **Total Classes:** 43 distinct categories of traffic signs (speed limits, warnings, prohibitions, etc.)
* **Train samples:** 31,367 images
* **Validation samples:** 7,842 images
* **Image dimensions:** Resized and normalized to 32x32x3 pixels.

---

🛠️ Model Architecture
The core of this system is a custom Deep Convolutional Neural Network (CNN) optimized with robust regularization layers to eliminate overfitting:
Conversion Blocks: 3 structural blocks using 3x3 filters with expanding channels (32 → 64 → 128) to learn spatial feature hierarchies.
Batch Normalization: Applied after every convolutional block to ensure internal covariate shift stability.
Dropout Regularization: Progressively tuned from 25% to 50% to prevent neural path co-dependency and overfitting.
Fully Connected Head: A dual-dense classifier block (512 → 256 → 43 neurons) utilizing Softmax activation for multi-class probability outputs.

🚀 High-Performance Training Pipeline
To protect the system against data-warping and cross-contamination bugs, a native tf.data.Dataset API pipeline was integrated:
Real-time Spatial Data Augmentation: Built explicitly using localized layers (RandomRotation, RandomTranslation, RandomZoom) protecting label vectors from unauthorized scaling modifications.
Optimizer: Adam Optimizer (initial Learning Rate = 0.001).
Callbacks:
EarlyStopping (patience=6, restoring best weights automatically).
ReduceLROnPlateau (dynamic learning rate halving based on val_loss stagnation).

📈 Training Results & Performance
The model converged seamlessly, unlocking rapid learning characteristics without any symptoms of validation divergence:
Training Accuracy: 99.73%
Validation Accuracy: 99.94%
Final Loss: 0.0091
Visualizing Optimization History
You can find the training evolution curves and confidence panel breakdowns inside the /results folder.
Sample Predictions	Confusion Matrix

💻 Installation & Usage
1. Clone the repository
Bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
2. Install dependencies
Bash
pip install -r requirements.txt
3. Run the notebook
Open the file inside the notebooks/ directory using Google Colab or Jupyter Notebook and execute all cells sequentially.

🛠️ Technologies Used
Python 3.12+
TensorFlow 2.x / Keras
NumPy & Pandas
Matplotlib & Seaborn
Scikit-Learn
