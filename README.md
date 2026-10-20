
# COVID-19 Chest CT Classifier

A three-class chest CT image classifier implemented using MobileNetV3-Large and PyTorch to classify CT slices as Normal, COVID-19 Positive or Non-informative.

The model was fine-tuned on 19,685 CT slices, achieving **99.59%** test accuracy on 2,956 held-out images. An interactive Streamlit application was developed for image classification.

<img width="1385" height="1136" alt="image" src="https://github.com/user-attachments/assets/868967d5-7356-4dba-b518-76cfd13170e9" />


**[Live Demo](https://covid19ctclassifier.streamlit.app/)**

## Tech Stack

Python, PyTorch, Torchvision, MobileNetV3-Large, Scikit-learn, Streamlit, Google Colab

## Dataset

[Preprocessed CT Scans for COVID-19 - Kaggle](https://www.kaggle.com/datasets/azaemon/preprocessed-ct-scans-for-covid19)

| Class | Images |
|---|---:|
| Normal (nCT) | 9,979 |
| COVID-19 Positive (pCT) | 4,001 |
| Non-informative (NiCT) | 5,705 |
| **Total** | **19,685** |

## Model Implementation

- Used ImageNet-pretrained MobileNetV3-Large with a modified three-class classification head.
- Fine-tuned the model for 25 epochs using PyTorch.
- Applied image augmentation, resizing and normalization during preprocessing.
- Evaluated performance using precision, recall, F1-score and a confusion matrix.

## Results

| Metric | Result |
|---|---:|
| Validation accuracy | 99.22% |
| Test accuracy | **99.59%** |
| Test images | 2,956 |
| Correct predictions | 2,944 |

## Model Evaluation
<img width="579" height="482" alt="image" src="https://github.com/user-attachments/assets/033ee5f8-3af0-4e2f-864a-53d5f65aa719" />
<img width="590" height="368" alt="image" src="https://github.com/user-attachments/assets/206c5ece-7c2c-4704-8aa5-ef64c274d234" />
<img width="590" height="368" alt="image" src="https://github.com/user-attachments/assets/05c1d9bc-8264-4931-8906-d67fede6a5e0" />




## Application

The deployed application accepts CT image uploads and displays the predicted class with probabilities for all three categories.

**[Try the Classifier](https://covid19ctclassifier.streamlit.app/)**

<img width="780" height="433" alt="image" src="https://github.com/user-attachments/assets/44c59717-6533-43e3-a216-3052b346a1cc" />

## Limitations

Evaluation was performed using a slice-level split, which may introduce patient-level data leakage. Results have not been externally validated. This project is intended for educational purposes, not clinical diagnosis.

## References

- [Original classifier implementation](https://github.com/chrispathway/covid-ct-classifier)
- [CT Scan Dataset - Kaggle](https://www.kaggle.com/datasets/azaemon/preprocessed-ct-scans-for-covid19)

Based on the referenced implementation, with model training, evaluation and Streamlit deployment completed as part of this project.
