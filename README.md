# Room Occupancy Estimation Using Machine Learning

## 1. Problem Statement

The objective of this project is to estimate the number of people occupying a room using environmental sensor measurements.

This project implements and compares five machine learning classification algorithms to identify the model that provides the best performance in predicting room occupancy.

An interactive Streamlit dashboard is also developed to allow users to upload test datasets, select trained machine learning models and evaluate their performance.

## 2. Dataset Description

**Dataset Name:** UCI Room Occupancy Estimation

**Dataset Source:** [UCI Machine Learning Repository – Room Occupancy Estimation](https://archive.ics.uci.edu/static/public/864/Occupancy_Estimation.csv)

**Dataset DOI:** 10.24432/C5P605

The dataset contains environmental sensor readings collected to estimate the number of people occupying a room.

| Dataset Attribute | Description |
|---|---|
| Total Instances | 10,129 |
| Total Original Columns | 19 |
| Input Features Used | 16 |
| Target Variable | Room_Occupancy_Count |
| Number of Classes | 4 |
| Problem Type | Multiclass Classification |

### Input Features

The following 16 sensor measurements were used:

- S1_Temp, S2_Temp, S3_Temp, S4_Temp
- S1_Light, S2_Light, S3_Light, S4_Light
- S1_Sound, S2_Sound, S3_Sound, S4_Sound
- S5_CO2
- S5_CO2_Slope
- S6_PIR
- S7_PIR

The Date and Time columns were excluded from model training.

### Target Variable

The target variable is `Room_Occupancy_Count`, which represents four occupancy levels:

| Target Value | Occupancy Class |
|---|---|
| 0 | Empty |
| 1 | One Person |
| 2 | Two People |
| 3 | Three People |

The dataset was checked for missing values and duplicate records before model development.

## 3. Data Preprocessing and Model Development

The dataset was divided into training and testing sets using an 80:20 split.

- Training data: 80%
- Testing data: 20%
- Random state: 42
- Stratified sampling was used to maintain class proportions.

Feature scaling was applied using StandardScaler for Logistic Regression, KNN, Gaussian Naive Bayes and SVM through Scikit-learn pipelines.

Decision Tree was implemented without feature scaling.

All five models were trained using the same training and testing split. The trained models were saved as Joblib files for subsequent evaluation through the Streamlit application.

## 4. Machine Learning Models Implemented

The following five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## 5. Model Performance Evaluation

The models were evaluated using six performance metrics:

- **Accuracy:** Measures the proportion of correctly classified observations.
- **AUC:** Measures the model's ability to distinguish between classes using the One-vs-Rest approach.
- **Precision:** Measures the proportion of positive predictions that are correct.
- **Recall:** Measures the proportion of actual observations correctly identified.
- **F1 Score:** Harmonic mean of precision and recall.
- **MCC:** Matthews Correlation Coefficient, which evaluates classification quality considering all confusion matrix categories.

Precision, Recall and F1 Score are calculated using macro averaging. AUC uses macro-averaged One-vs-Rest calculation.

### 5.1 Model Comparison Table

The following results were obtained from the original test dataset.

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9946 | 0.9996 | 0.9850 | 0.9803 | 0.9822 | 0.9834 |
| Decision Tree | 0.9961 | 0.9929 | 0.9871 | 0.9874 | 0.9871 | 0.9880 |
| KNN | 0.9951 | 0.9979 | 0.9823 | 0.9845 | 0.9833 | 0.9850 |
| Naive Bayes | 0.9615 | 0.9931 | 0.8809 | 0.8880 | 0.8821 | 0.8844 |
| **Support Vector Machine** | **0.9970** | **0.9996** | **0.9898** | **0.9925** | **0.9911** | **0.9910** |

### 5.2 Observations on Individual Models

**1. Logistic Regression**

Logistic Regression achieved an accuracy of 99.46% and a macro F1 score of 0.9822. It performed well across all occupancy classes. However, its recall for the Three People class was slightly lower compared to the other classes.

**2. Decision Tree**

Decision Tree achieved an accuracy of 99.61% and a macro F1 score of 0.9871. It provided strong classification performance across all four occupancy classes and maintained a good balance between precision and recall.

**3. K-Nearest Neighbors (KNN)**

KNN achieved an accuracy of 99.51% and a macro F1 score of 0.9833. It performed consistently across the occupancy classes, with particularly strong recall for the Two People class.

**4. Gaussian Naive Bayes**

Naive Bayes achieved an accuracy of 96.15% and a macro F1 score of 0.8821. It performed well for the Empty and One Person classes but showed comparatively lower precision and recall for the Two People and Three People classes.

**5. Support Vector Machine (SVM)**

SVM achieved an accuracy of 99.70%, a macro F1 score of 0.9911 and an MCC score of 0.9910. It demonstrated the strongest overall classification performance among the five evaluated models.

### 5.3 Overall Best Model

**Support Vector Machine (SVM) is the overall best-performing model.**

It achieved:

- Highest Accuracy: 0.9970
- Highest Precision: 0.9898
- Highest Recall: 0.9925
- Highest F1 Score: 0.9911
- Highest MCC: 0.9910
- AUC: 0.9996, tied with Logistic Regression for the highest score.

Based on these results, SVM demonstrated the best overall performance for room occupancy estimation on the selected test dataset.

## 6. Streamlit Application

An interactive web application was developed using Streamlit to evaluate the saved machine learning models.

### Application Features

- Upload a CSV test dataset containing sensor features and the target variable.
- Preview uploaded dataset records.
- View dataset information and occupancy classes.
- Select one of the five trained machine learning models.
- Calculate Accuracy, AUC, Precision, Recall, F1 Score and MCC.
- Display a confusion matrix.
- Display a classification report.
- Compare the performance of all five models.

The application loads the previously trained models from the `model` directory. It evaluates the uploaded test dataset without retraining the models.

### Live Application

[**Open Room Occupancy Estimation – Streamlit Application**](https://sanjiv6213-ml-assignment-2-app-frquqt.streamlit.app/)

## 7. GitHub Repository

The complete source code, training notebook, saved models, test dataset and application dependencies are available in the GitHub repository.

[**View GitHub Repository – ML Assignment 2**](https://github.com/sanjiv6213/ML-Assignment-2/tree/main)

## 8. Project Structure

```text
ML-Assignment-2/
│
├── app.py
├── ML_Assignment_2.ipynb
├── README.md
├── requirements.txt
├── Occupancy_Estimation.csv
├── test_data.csv
├── model_comparison.csv
│
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
```

## 9. Technologies Used

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Model Persistence:** Joblib
- **Data Visualization:** Matplotlib, Seaborn
- **Web Application:** Streamlit
- **Development Environment:** Google Colab, Visual Studio Code
- **Version Control:** Git and GitHub
- **Deployment:** Streamlit Community Cloud

## 10. Conclusion

This project demonstrates the implementation and comparison of five machine learning classification algorithms for estimating room occupancy using environmental sensor measurements.

The models were evaluated using six performance metrics. Among the five algorithms, Support Vector Machine achieved the best overall performance, while Gaussian Naive Bayes showed comparatively lower performance.

The Streamlit dashboard provides an interactive interface to evaluate the saved models using uploaded test data without requiring model retraining.

The project demonstrates the application of machine learning techniques for sensor-based room occupancy estimation.
