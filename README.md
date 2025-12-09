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
├── data/                     # Download from Kaggle https://www.kaggle.com/competitions/cassava-disease-classification/data
│   └── cassava-disease-classification/
│       ├── train/
│       ├── test/
│       └── extraimages/
│
├── notebooks/
│   └── cassava-disease-classification.ipynb
│
├── src/
│   ├── dataset.py
│   ├── training.py
│   ├── evaluate.py
│   ├── inference.py
│   └── utils.py
│
├── models/                  # Generated after model training
│   └── *.pth                # (not included – too large)
│
├── reports/                 # training curves, confusion matrices and classification reports.
│   └── *.png
│
├── submissions/
│   └── *.csv                # Generated kaggle submission files
│ 
├── test/                    # Unit tests for dataset
│   └── test_datasets.py

```

---

## Model Weights & Reports

To keep the repository lightweight, trained model weights and report files i.e confusion matrices are stored externally.

[**Google Drive Download Link**](https://drive.google.com/drive/folders/1Mk_ivEI9exi00SzLjb_CwMaU4oeIuve-?usp=drive_link)  

This folder contains:
- `models/*.pth` — fine-tuned ConvNeXt, EfficientNet-B4, DenseNet121
- `reports/*.png` — confusion matrices & classification reports. 
- `submissions/*.csv` — kaggle submission files. 

After downloading, place the files into there respective folders.

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

You can train **all models**, a **single model**, or **multiple selected models** directly from the terminal.

```bash
# Train all models
python -m src.training --models all

# Train a single model: ConvNeXt/EfficientNet-B4/DenseNet121
python -m src.training --models ConvNeXt

# Train multiple specific models: DenseNet121 and ConvNeXt
python -m src.training --models DenseNet121 ConvNeXt

# Enable MixUp/CutMix augmentation
python -m src.training --models ConvNeXt --mixup

# Enable early stopping
python -m src.training --models EfficientNet-B4 --early-stop

# Override the default number of epochs
python -m src.training --models ConvNeXt --epochs 20

```

You may also train directly from the notebook:

```text
notebooks/3_cassava-disease-classification.ipynb
```

---

## **Evaluation**

```bash
# Single model evaluation: convnext/efficientnet/densenet
python -m src.evaluate --model convnext

# Ensemble evaluation
python -m src.evaluate --ensemble
```

---

## **Inference**

```bash
# Single model inference: convnext/efficientnet/densenet
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
<table> <tr> 
<td width="50%" valign="top"> 
  <h3>Emmanuel Kirui Barkacha</h3> 
  <ul> 
    <li><strong>Email:</strong> <a href="mailto:ebarkacha@aimsammi.org">ebarkacha@aimsammi.org</a>
    </li> <li><strong>LinkedIn:</strong> <a href="https://www.linkedin.com/in/emmanuel-kirui-barkacha-493807294">linkedin.com/in/emmanuel-kirui-barkacha</a></li> <li><strong>GitHub:</strong> <a href="https://github.com/ekbarkacha">github.com/ekbarkacha</a></li> </ul> 
</td>
<td width="50%" valign="top">
  <h3> Rabemanantsoa Andriamianja Tiana</h3>
  <ul>
    <li><strong>Email:</strong> <a href="mailto:arabemanantsoa@aimsammi.org">arabemanantsoa@aimsammi.org</a></li>
    <li><strong>LinkedIn:</strong> <a href="https://www.linkedin.com/in/rabemanantsoa-andriamianja">linkedin.com/in/rabemanantsoa-andriamianja</a></li>
    <li><strong>GitHub:</strong> <a href="https://github.com/Mianja-Tiana">github.com/Mianja-Tiana</a></li>
  </ul>
</td>
</tr> </table>

---

## **License**

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details. 