# **Cassava Disease Classification (Kaggle Competition)**

*A deep learning pipeline for fine-grained cassava leaf disease classification using transfer learning.*

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF)

---

## **Project Overview**

Cassava is a vital food crop for millions of people across Sub-Saharan Africa. However, several diseases significantly reduce cassava yields and are difficult to diagnose manually due to their visual similarity.

This project solves this challenge by building a **deep learning–based image classification system** that automatically detects cassava leaf diseases using **transfer learning on modern CNN architectures**.

The system classifies each leaf image into **five categories**:

| Class   | Description                  |
| ------- | ---------------------------- |
| CBB     | Cassava Bacterial Blight     |
| CBSD    | Cassava Brown Streak Disease |
| CGM     | Cassava Green Mite           |
| CMD     | Cassava Mosaic Disease       |
| Healthy | No disease                   |

**Kaggle Competition:**
[https://www.kaggle.com/competitions/cassava-disease-classification](https://www.kaggle.com/competitions/cassava-disease-classification)

---

## Project Report

The full project report, including methodology, experiments and analysis, is available here:

**[View Full Project Report (PDF)](docs/emmanuel_tiana_cassava_disease_classification_report.pdf)**

---

## **Models Explored**

We trained and evaluated the following pretrained CNN architectures:

* **DenseNet121**
* **EfficientNet-B4**
* **ConvNeXt-Tiny** → **Best performing model**

All models were fine-tuned using **PyTorch** on a **NVIDIA Tesla P100 GPU** via Kaggle Notebooks .

---

## **Final Results**

| Model             | Validation Accuracy | Macro F1   | Local Test Accuracy |
| ----------------- | ------------------- | ---------- | ------------------- |
| DenseNet121       | 89.93%              | 0.8507     | 88.87%              |
| EfficientNet-B4   | 91.71%              | 0.8720     | 90.11%              |
| **ConvNeXt-Tiny** | **91.87%**          | **0.8882** | **90.81%**          |

### **Kaggle Leaderboard Performance**

* **Public Leaderboard Accuracy:** **91.456%**
* **Leaderboard Rank:** **Top Position**
* **Final Model:** **ConvNeXt-Tiny**
* **Evaluation Metric:** Macro F1-score

These results demonstrate strong generalization and robustness against class imbalance .

---

## **Project Structure**

```bash
cassava-disease-classification/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── cassava-disease-classification/
│       ├── train/
│       ├── test/
│       └── extraimages/
│
├── notebooks/
│   └── 3_cassava-disease-classification.ipynb
│
├── src/
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── utils.py
│
├── models/
│   └── ConvNeXt.pth
│
├── reports/
│   └── ConvNeXt_confusion_matrix_and_classification_report.png
│
├── submissions/
│   └── submission_convnext.csv
```

---

## **Dataset Description**

* **Total Images:** 21,367
* **Labeled Training Images:** 9,436
* **Unlabeled Test Images:** 12,595
* **Image Source:** Smartphone field data (Uganda)
* **Annotations:** Verified by agricultural experts at NaCRRI
* **Class Imbalance:** CMD and CBSD dominate the dataset

*The dataset is not included in this repository but you can download it from [kaggle](https://www.kaggle.com/competitions/cassava-disease-classification/data)* .

---

## **Data Preprocessing & Augmentation**

* Image resizing: **380 × 380**
* Normalization: **ImageNet mean & std**
* Augmentations:

  * Random horizontal/vertical flips
  * Random rotations
  * Color jitter
  * Random cropping and resizing

These steps significantly improved class generalization and reduced overfitting.

---

## **Training Configuration**

* **Framework:** PyTorch
* **Loss Function:** Cross-Entropy with **Label Smoothing**
* **Optimizer:** AdamW
* **LR Scheduler:** Cosine Annealing
* **Train/Val/Test Split:** 80% / 10% / 10% (Stratified)
* **Metrics:** Accuracy, Macro F1-Score
* **Hardware:** Tesla P100 GPU (Kaggle)

---

## **Installation**

### Clone this repository

```bash
git clone https://github.com/ekbarkacha/cassava-disease-classification.git
cd cassava-disease-classification
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Download Kaggle Dataset

```bash
kaggle competitions download -c cassava-disease-classification
unzip cassava-disease-classification.zip -d data/
```

---

## **Training the Model**

```bash
# Train single ConvNeXt model
python -m src.train --model convnext

# Train ensemble models
python -m src.train --ensemble
```

You may also train directly from the notebook:

```text
notebooks/3_cassava-disease-classification.ipynb
```

---

## **Evaluation**

```bash
# Single model evaluation
python -m src.evaluate --model convnext

# Ensemble evaluation
python -m src.evaluate --ensemble
```

---

## **Inference**

```bash
# Single model inference
python -m src.inference --model convnext

# Ensemble inference
python -m src.inference --ensemble
```

---

## **Future Improvements**

* Test-Time Augmentation (TTA)
* Pseudo-labeling using unlabeled images
* Web deployment FastAPI

---

## **Authors**

* **Emmanuel Kirui Barkacha** – [ebarkacha@aimsammi.org](mailto:ebarkacha@aimsammi.org)
* **Rabemanantsoa Andriamianja Tiana** – [arabemanantsoa@aimsammi.org](mailto:arabemanantsoa@aimsammi.org)

---

## **License**

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details. 