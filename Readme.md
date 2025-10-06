Diabetes Prediction using Support Vector Machine (SVM)
This project predicts the onset of diabetes based on diagnostic medical measurements using a Support Vector Machine (SVM) classifier. The goal is to build a model that can accurately identify whether a patient has diabetes.

## Dataset
This project uses the PIMA Indians Diabetes Dataset.

This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. It contains several medical predictor variables and one target variable, Outcome. Predictor variables include the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.

Source: Kaggle: PIMA Indians Diabetes Dataset

Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age

Target Variable: Outcome (0 for non-diabetic, 1 for diabetic)

## Getting Started
To get a local copy up and running, follow these simple steps.

### Prerequisites
You'll need Python 3.x and the following libraries installed:

NumPy

Pandas

Scikit-learn

Matplotlib

Seaborn

### Installation
Clone the repo:

Bash

git clone https://github.com/am-prasad/Diabetics_prediction.git
Install the required packages:

Bash

pip install numpy pandas scikit-learn matplotlib seaborn
### How to Use
Open the diabetics_prediction.ipynb notebook in Jupyter Notebook, JupyterLab, or Google Colab and run the cells sequentially to see the analysis and model results.

## Methodology
The project follows a standard machine learning workflow:

Data Loading & Exploration: The dataset is loaded using Pandas, and an initial exploratory data analysis (EDA) is performed to understand its structure, distributions, and correlations.

Data Preprocessing: The data is cleaned and prepared for modeling. This includes standardizing the features using StandardScaler to ensure that all variables contribute equally to the model's performance.

Train-Test Split: The dataset is split into a training set (80%) and a testing set (20%) to evaluate the model on unseen data.

Model Training: A Support Vector Machine (SVM) classifier with a linear kernel (SVC(kernel='linear')) is trained on the preprocessed training data.

Model Evaluation: The model's performance is evaluated on the test set using several metrics:

Accuracy Score: The overall percentage of correct predictions.

Confusion Matrix: A table showing the number of true positives, true negatives, false positives, and false negatives.

Classification Report: A detailed report including precision, recall, and F1-score for each class.

## Results
The trained SVM model achieved an accuracy of ~77% on the test data. The classification report provides a detailed look into its predictive power for both diabetic and non-diabetic classes, showing a good balance between precision and recall.

For more details, please refer to the evaluation section in the notebook.

## License
Distributed under the MIT License. See LICENSE for more information.
