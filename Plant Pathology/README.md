# Plant Pathology – VGG19 Classification

## Project Overview
This project detects and classifies plant leaf diseases (Healthy, Rust,and Powdery) using a Convolutional Neural Network (CNN) with a pre‑trained VGG19 model (transfer learning).

## Dataset :
Provide the dataset link here: [https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset/data"  # optional, change or remove].

## Model Details:
-   Base Model: VGG19 (ImageNet weights, top removed)
-   Input Size: 224×224×3
-   Data Generators: ImageDataGenerator for training validation, and testing.
-   Loss: categorical_crossentropy
-   Optimizer: Adam (default learning rate)
-   Metrics: Accuracy

## Training Results: 


| Metric              |   Value |
|---------------------|--------:|
| Training Accuracy   | 0.9220 |
| Validation Accuracy | 0.9333 |
| Training Loss       | 0.2641 |
| Validation Loss     | 0.1923 |

These results were obtained after 4 epochs.

## How to Run:
1.  Clone the repository.
2.  Place the dataset in the specified folder and update train_dir,
    val_dir, and test_dir paths.
3.  Run the Jupyter notebook to train or evaluate the model.
---
Add screenshots or a mini video demonstrating model training and
predictions here. 

Befor Prediction:

<img width="352" height="226" alt="upload file" src="https://github.com/user-attachments/assets/41aa6720-2a78-4052-be46-fec86b368d91" />

After predection:

<img width="608" height="370" alt="plant pathology2" src="https://github.com/user-attachments/assets/0ce338f4-a5e2-4f9f-8976-84a68eaacb8e" />
