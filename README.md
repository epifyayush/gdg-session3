# Diabetes Prediction Using Machine Learning

## Overview
This repository contains a machine learning model that predicts diabetes outcomes based on patient data. The model is built using Python and employs a Support Vector Machine (SVM) classifier to classify patients as diabetic or non-diabetic.

## Dataset
The dataset used for training and testing is loaded from `diabetes (2).csv`. It includes various medical measurements such as glucose levels, BMI, and insulin levels, with an outcome column indicating whether the patient is diabetic (`1`) or non-diabetic (`0`).

## Features Used
The model uses the following features:
- Glucose level
- Blood pressure
- Skin thickness
- Insulin level
- BMI (Body Mass Index)
- Diabetes Pedigree Function
- Age
- Pregnancies

The `Outcome` column is used as the target variable.

## Dependencies
Ensure you have the following Python libraries installed:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Model Training & Evaluation
1. Load the dataset and check for missing values.
2. Perform exploratory data analysis, including visualizing the distribution of glucose levels.
3. Preprocess data using `StandardScaler`.
4. Split the data into training (80%) and testing (20%) sets.
5. Train a Support Vector Machine (SVM) model with a linear kernel.
6. Evaluate model performance using accuracy score, classification report, and confusion matrix.

## Running the Code
Execute the script using Python:

```bash
python machinelearning(1).py
```

## Example Prediction
The script includes an example prediction for a new patient:
```python
new_patient = np.array([[2, 250, 30, 45, 100, 35.0, 0.5, 55]])
new_patient_scaled = scalar.transform(new_patient)
prediction = model.predict(new_patient_scaled)
print("Predicted Diabetes outcome:", "Diabetic" if prediction[0] == 1 else "Not Diabetic")
```

## Results
The model prints the accuracy percentage along with classification metrics:
- **Accuracy:** Measures how well the model predicts diabetes outcomes.
- **Confusion Matrix:** Shows true positives, false positives, true negatives, and false negatives.
- **Classification Report:** Includes precision, recall, and F1-score for both classes.

## Contributing
Feel free to fork this repository and improve the model by experimenting with different classifiers, hyperparameters, or feature engineering techniques.

## License
This project is open-source and available under the MIT License.

