# Room Occupancy Estimation Using Machine Learning

## 1. Problem Statement

The objective of this project is to estimate the number of people occupying a room using environmental sensor measurements.

Five machine learning classification algorithms are implemented and evaluated to identify the model that performs best in predicting room occupancy.

The project also provides an interactive Streamlit dashboard where users can upload test data, select a trained model and view its evaluation results.

## 2. Dataset Description

**Dataset:** UCI Room Occupancy Estimation

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/static/public/864/Occupancy_Estimation.csv)

The dataset contains environmental sensor readings collected for estimating room occupancy.

| Dataset attribute | Description |
|---|---|
| Total instances | 10,129 |
| Input features | 16 sensor variables |
| Target variable | Room_Occupancy_Count |
| Number of classes | 4 |
| Problem type | Multiclass classification |

The 16 input features include temperature, light, sound, CO2, CO2 slope and PIR sensor measurements.

The target variable represents four occupancy levels:

- 0 – Empty
- 1 – One Person
- 2 – Two People
- 3 – Three People

The Date and Time columns were excluded from the model input.

## 3. GitHub Repository Link

**GitHub Repository:** [Add your GitHub repository link here]

## 4. Machine Learning Models Used

The following five classification algorithms were implemented using the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The dataset was divided into 80% training data and 20% testing data using a stratified train-test split with random state 42.

Feature scaling was applied through pipelines for Logistic Regression, KNN, Gaussian Naive Bayes and SVM. Decision Tree was implemented without feature scaling.

### 4.1 Model Performance Comparison

The following metrics were calculated on the original test dataset.

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9946 | 0.9996 | 0.9850 | 0.9803 | 0.9822 | 0.9834 |
| Decision Tree | 0.9961 | 0.9929 | 0.9871 | 0.9874 | 0.9871 | 0.9880 |
| KNN | 0.9951 | 0.9979 | 0.9823 | 0.9845 | 0.9833 | 0.9850 |
| Naive Bayes | 0.9615 | 0.9931 | 0.8809 | 0.8880 | 0.8821 | 0.8844 |
| Support Vector Machine | **0.9970** | **0.9996** | **0.9898** | **0.9925** | **0.9911** | **0.9910** |

*Precision, Recall and F1 Score are macro-averaged. AUC is calculated using the One-vs-Rest approach with macro averaging.*

### 4.2 Observations on Model Performance

**Logistic Regression**

Logistic Regression achieved 99.46% accuracy and a macro F1 score of 0.9822. It performed well across all occupancy classes, although its recall for the Three People class was slightly lower.

**Decision Tree**

Decision Tree achieved 99.61% accuracy and a macro F1 score of 0.9871. It provided strong classification performance across all four occupancy classes.

**K-Nearest Neighbors (KNN)**

KNN achieved 99.51% accuracy and a macro F1 score of 0.9833. It performed consistently across the occupancy classes, with particularly strong recall for the Two People class.

**Gaussian Naive Bayes**

Naive Bayes achieved 96.15% accuracy and a macro F1 score of 0.8821. It performed well for the Empty and One Person classes but showed comparatively lower precision and recall for the Two People and Three People classes.

**Support Vector Machine (SVM)**

SVM achieved the highest accuracy of 99.70%, a macro F1 score of 0.9911 and an MCC score of 0.9910. It demonstrated the strongest overall classification performance among the five models.

### 4.3 Overall Winner

**Support Vector Machine (SVM)** is the overall winner for this dataset.

It achieved the highest accuracy, precision, recall, F1 score and MCC among the five evaluated models. Its AUC score was also tied for the highest value at 0.9996.

Therefore, SVM demonstrated the best overall performance on the original test dataset.

## 5. Streamlit Application

The project includes an interactive Streamlit dashboard with the following features:

- CSV test dataset upload.
- Machine learning model selection.
- Dataset overview and occupancy distribution.
- Evaluation metrics: Accuracy, AUC, Precision, Recall, F1 Score and MCC.
- Confusion matrix.
- Classification report.
- Performance comparison across all five models.
- Dataset exploration and CSV download.

The application loads previously trained models and evaluates uploaded test data without retraining.

**Live Streamlit Application:** [Add your deployed Streamlit application link here]

## 6. Project Structure

```text
Room_Occupancy_ML/
│
├── app.py
├── train_models.ipynb
├── requirements.txt
├── README.md
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

## 7. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Google Colab
- Visual Studio Code

## 8. Conclusion

This project demonstrates the implementation and comparison of five machine learning classification algorithms for room occupancy estimation.

The results show that Support Vector Machine achieved the best overall performance, while Gaussian Naive Bayes had comparatively lower performance on this dataset.

The Streamlit application provides an interactive interface for evaluating the saved models using uploaded test data.
