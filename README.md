# README for Diabetes Prediction Model Comparison  

## Overview  
This project aims to compare the performance of various machine learning models for predicting diabetes. The models evaluated include K-Nearest Neighbors (KNN), Decision Trees (standard and with weighted processing), Gradient Boosting Decision Trees, Random Forests, and Logistic Regression. The results are compared based on key evaluation metrics to determine which model performs best for this task.  

## Environment  
- **Python Version**: 3.12.0  
- **Libraries**:  
  - `scikit-learn`  
  - `numpy`  
  - `pandas`  

## Dataset  
The dataset used for this experiment consists of **300,000 samples** with **22 feature fields**:  
- Sample ID  
- High Blood Pressure (HighBP)  
- High Cholesterol (HighChol)  
- Cholesterol Check (CholCheck)  
- Body Mass Index (BMI)  
- Smoker Status (Smoker)  
- Stroke History  
- Heart Disease History  
- Frequent Physical Activity in the Last 30 Days  
- Daily Fruit Consumption  
- Daily Vegetable Consumption  
- Alcohol Use  
- Any Healthcare Access  
- Concern for Medical Costs  
- General Health Status  
- Mental Health Status  
- Physical Health Status  
- Difficulty Walking/Stairs  
- Gender  
- Age  
- **Target**: The prediction target, where values are originally 0 (not diabetic), 1 (diabetic), and 2 (prediabetic). For easier processing, the target variable is binarized to 0 (not diabetic) and 1 (diabetic).  

### Feature Distribution  
As shown in **Figure 3-1**, features such as HighBP, HighChol, CholCheck, and Smoker are binary features, while the others are non-binary discrete features. Some feature distributions are imbalanced, which may affect subsequent model training and evaluation.  

## Feature Selection  
To identify the optimal feature set for diabetes prediction, various methods were used to analyze and filter the attributes. Each attribute's correlation with the target variable was assessed to determine which attributes are critical for model accuracy and stability.   

1. **Visualization**: The relationship between feature values and the target variable was visualized using stacked bar charts (see **Figure 3-2**) to intuitively observe the correlation.  
   
2. **Correlation Heatmap**: A correlation heatmap (see **Figure 3-3**) was generated using Pearson correlation coefficients and mutual information scores to classify features into high correlation, low correlation, and potentially redundant features.  

3. **Feature Selection**: L1 regularization was employed for feature selection, followed by L2 regularization for cross-validation. Random Forest and Decision Tree feature scoring were combined with earlier correlation analysis results to finalize the input features. The selected features include HighChol, BMI, and others.  

## Data Preprocessing  
1. **Train-Test Split**: The dataset was split into training and testing sets using the `train_test_split` function, with a ratio of 80:20. The training set is used for model training, while the testing set is used to evaluate the final model's predictive performance.  

2. **Standardization**: Feature data was standardized using `StandardScaler` to ensure all features are on the same scale. The standardized data was then converted back to a DataFrame for further processing.  

3. **Handling Imbalanced Data**: Given the significant differences in many binary classification feature values, upsampling was chosen to address the data imbalance issue.  

## Models Evaluated  
1. **K-Nearest Neighbors (KNN)**: Known for its simplicity and effectiveness, especially with imbalanced datasets.  
2. **Decision Trees**: Classic decision-making models that can be prone to overfitting.  
3. **Weighted Decision Trees**: Modified decision tree model aimed at improving performance on imbalanced data.  
4. **Gradient Boosting Decision Trees**: An ensemble method that aims to improve model performance over standard decision trees.  
5. **Random Forests**: An ensemble of decision trees which enhances accuracy and reduces overfitting risks.  
6. **Logistic Regression**: A fundamental statistical model for binary classifications.  

## Key Metrics  
The following metrics were used to evaluate the models:  

- **F1 Score**: A measure of a model's accuracy that considers both precision and recall.  
- **Bias**: Measures the model's error due to overly simplistic assumptions in the learning algorithm.  
- **Variance**: Measures how much the model's predictions vary for different training sets.  
- **Accuracy**: The proportion of true results among the total number of cases examined.  

### Results Summary  

| Model                             | F1 Score | Bias  | Variance | Accuracy |  
|-----------------------------------|----------|-------|----------|----------|  
| KNN                               | 0.82     | 0.10  | 0.14     | 0.90     |  
| Decision Tree                     | 0.82     | 0.18  | 0.14     | 0.82     |  
| Weighted Decision Tree            | 0.82     | 0.18  | 0.14     | 0.82     |  
| Gradient Boosting Decision Tree   | 0.81     | 0.10  | 0.03     | 0.85     |  
| Random Forest                     | 0.86     | 0.09  | 0.04     | 0.87     |  
| Logistic Regression               | 0.81     | 0.16  | 0.05     | 0.84     |  

### Findings  
- **KNN** achieved the highest accuracy but had a slightly lower F1 score than Random Forest. Its bias was low, indicating reliable predictions but with moderate variance.  
- **Decision Trees** and their weighted version showed comparable F1 scores but faced higher bias and potential overfitting. Gradient Boosting provided better accuracy and lower variance compared to standard decision trees.  
- **Random Forest** outperformed all models in terms of F1 score, accuracy, and bias, showcasing excellent stability and generalization abilities.  
- **Logistic Regression** had a lower F1 score and higher bias but maintained good accuracy and low variance.  

## Conclusion  
The Random Forest model emerged as the best-performing model for diabetes prediction, balancing accuracy, F1 score, and variance effectively. The KNN model, while accurate, showed greater bias, and gradient boosting methods exhibited notable stability but with slightly lower scores.  

## Installation  
To run this project, clone the repository and install the required libraries using pip:  

```bash  
git clone [https://github.com/the10world/-.git]  
cd [https://github.com/the10world/-/tree/master]  
