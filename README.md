# **Machine Learning Projects Overview**

This repository contains three key projects that demonstrate practical applications of **machine learning** in text classification, clustering, and regression with regularization. Below, you'll find a detailed description of each project, the techniques used, and the expected outputs.

---

## **1. Text Classification: Spam Detection**
The `spam_detector` function classifies messages as **spam** or **ham** (not spam) using various machine learning models and text vectorization techniques.

### **Techniques and Tools**:
- **TF-IDF Vectorization**:
  - Converts text into numerical representations based on word relevance.
  - Includes unigrams and bigrams, filtered by a minimum frequency of 2.
- **Classification Models**:
  - **Logistic Regression**
  - **Multinomial Naive Bayes**
  - **Decision Tree Classifier**
  - **Linear Support Vector Classifier (SVC)**
- **Validation**:
  - Generates confusion matrices for model evaluation.
  - Identifies the best classifier based on minimal misclassification of spam as ham.

### **Outputs**:
1. Confusion matrices for each classifier on the validation dataset.
2. The best-performing classifier with minimal spam misclassification.
3. TF-IDF matrix for the test dataset.
4. Predicted labels for the test dataset using the best classifier.

---

## **2. Clustering: Article Grouping**
The `cluster_articles` function groups documents into clusters using **K-Means** and evaluates the clustering quality with dimensionality reduction and performance metrics.

### **Techniques and Tools**:
- **K-Means Clustering**:
  - Clusters document vectors into 10 groups based on similarity.
  - Parameters: `n_clusters=10`, `random_state=2`, `tol=0.05`, `max_iter=50`.
- **Principal Component Analysis (PCA)**:
  - Reduces data dimensionality to 10 components.
  - Captures key variance in data for improved clustering.
- **Metrics for Evaluation**:
  - **Completeness Score**: Measures how well true group labels are assigned.
  - **V-Measure**: Combines homogeneity and completeness for clustering evaluation.

### **Outputs**:
1. Number of observations per cluster before and after PCA.
2. Variance explained by the first PCA component.
3. Completeness and V-Measure metrics for clustering before and after PCA.

---

## **3. Regression with Regularization**
The `regression` function applies **Ridge**, **Lasso**, and **Elastic Net** regression models to predict housing prices based on economic, cost, and construction variables.

### **Techniques and Tools**:
- **Ridge Regression**:
  - Penalizes large coefficients to prevent overfitting.
- **Lasso Regression**:
  - Shrinks coefficients, forcing some to become zero for feature selection.
- **Elastic Net Regression**:
  - Combines penalties from Ridge and Lasso.
  - Parameters: `alphas=np.logspace(-4, -1, 4)`, `l1_ratio=np.arange(0.6, 1, 0.1)`.

### **Outputs**:
1. Optimal `alpha` (regularization strength) for all models.
2. Optimal `l1_ratio` for Elastic Net.
3. Predictions for test data, rounded to two decimal places.
4. Coefficient values for features with significant impact:
   - For Ridge and Elastic Net: `abs(coef) > 0.001`.
   - For Lasso: `coef != 0`.

---

## **How to Run the Code**
1. Ensure you have Python installed with all required libraries (`pandas`, `scikit-learn`, etc.).
2. Load the datasets as instructed for each project.
3. Run the corresponding function:
   - `spam_detector` for text classification.
   - `cluster_articles` for clustering.
   - `regression` for regression with regularization.

### **Dependencies**
- Python >= 3.7
- pandas
- scikit-learn
- numpy

### **Author**
This repository was developed to demonstrate the use of machine learning models in various domains. Contributions and suggestions are welcome!
