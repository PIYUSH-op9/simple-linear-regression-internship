# Simple Linear Regression: Diabetes Progression Analysis

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Library](https://img.shields.io/badge/library-scikit--learn-orange)
![License](https://img.shields.io/badge/license-MIT-green)

A comprehensive machine learning workflow demonstrating Ordinary Least Squares (OLS) Simple Linear Regression to model and predict disease progression in diabetes patients using Body Mass Index (`bmi`).

---

## 📌 Executive Summary

Predicting disease progression is a key challenge in healthcare analytics. This project explores the mathematical and empirical relationship between a single physical indicator—Body Mass Index (BMI)—and a quantitative measure of diabetes progression recorded one year after baseline. 

Using the standard **Scikit-Learn Diabetes Dataset**, this project provides a clear template for:
1. Data pre-processing and feature selection.
2. Building and fitting a single-variable Linear Regression model.
3. Calculating key evaluation metrics ($MAE$, $RMSE$, $R^2$).
4. Interpreting regression coefficients in a real-world biomedical context.

---

## 📊 Dataset Description

The dataset used in this analysis originates from Bradley Efron, Trevor Hastie, Iain Johnstone, and Robert Tibshirani (2004) *"Least Angle Regression"*.

* **Total Samples:** 442 patients
* **Features:** 10 baseline variables (age, sex, body mass index, average blood pressure, and six blood serum measurements)
* **Target Variable ($y$):** Quantitative measure of disease progression one year after baseline (ranges from $25$ to $346$)
* **Selected Feature ($X$):** Body Mass Index (`bmi`), standardized/centered according to Scikit-Learn standards

---

## 🛠 Project Structure & Requirements

### Dependencies
Ensure you have the following packages installed:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Suggested Project Layout
```text
simple-linear-regression-internship/
│
├── data/
│   └── README.md
├── notebooks/
│   └── simple_linear_regression.ipynb
├── src/
│   └── train.py
├── README.md
└── requirements.txt
```

---

## ⚙️ Model Construction & Workflow

### 1. Data Splitting
The dataset is split into training and testing sets to evaluate generalization on unseen data:
* **Train Set:** 80% (353 samples)
* **Test Set:** 20% (89 samples)
* **Random State:** 42 (ensures reproducibility)

### 2. Model Architecture
Ordinary Least Squares (OLS) minimizes the Residual Sum of Squares (RSS) between the observed responses in the dataset and the responses predicted by the linear approximation.

$$\hat{y} = \beta_0 + \beta_1 X$$

* **Intercept ($\beta_0$):** $\approx 152.00$
* **Slope / Coefficient ($\beta_1$):** $\approx 998.58$

---

## 📈 Evaluation & Key Results

The model was evaluated on the unseen test set (89 samples).

| Metric | Formula | Value | Interpretation |
| :--- | :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | $\frac{1}{n} \sum \|y - \hat{y}\|$ | **52.26** | Average magnitude of errors in progression units. |
| **Root Mean Squared Error (RMSE)** | $\sqrt{\frac{1}{n} \sum (y - \hat{y})^2}$ | **63.73** | Penalizes larger prediction errors more heavily. |
| **Coefficient of Determination ($R^2$)** | $1 - \frac{SS_{res}}{SS_{tot}}$ | **0.233** | The model explains **~23.3%** of the variance in the target. |

---

## 🔍 Key Insights & Limitations

1. **Positive Correlation:** The positive coefficient ($\beta_1 = 998.58$) indicates that higher BMI values correlate with accelerated disease progression.
2. **Variance Explanation:** An $R^2$ score of $0.233$ shows that while BMI is a statistically significant feature, a single variable is insufficient to account for the complex physiological factors driving diabetes progression.
3. **Future Improvements:**
   * **Multiple Linear Regression:** Incorporate blood serum measurements (`s1` - `s6`) and blood pressure (`bp`).
   * **Feature Scaling & Transformation:** Apply non-linear transformations or polynomial features.
   * **Regularization:** Test Ridge and Lasso models to mitigate potential overfitting when introducing multiple features.

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/simple-linear-regression-internship.git
   cd simple-linear-regression-internship
   ```

2. **Run the script:**
   ```bash
   python src/train.py
   ```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Task 2: Classification on Imbalanced Data

### Objective
Build a classification model and evaluate it properly on imbalanced data.

### Dataset
Breast Cancer Wisconsin dataset from Scikit-learn. The malignant class is treated as the positive class (label 1).

### Work Completed
- Loaded the dataset and examined class distribution.
- Trained a Logistic Regression classifier using feature scaling and class weighting.
- Evaluated the model using a confusion matrix, accuracy, precision, recall, and F1-score.
- Compared classification thresholds of 0.5 and 0.3.
- Explained why accuracy alone can be misleading when classes are imbalanced.

### Results

| Metric | Threshold 0.5 | Threshold 0.3 |
|---|---:|---:|
| Accuracy | 97.4% | 96.5% |
| Precision | 97.6% | 95.2% |
| Malignant Recall | 95.2% | 95.2% |
| F1-score | 96.4% | 95.2% |

### Conclusion
In this comparison, the threshold of 0.5 achieved higher accuracy, precision, and F1-score, while malignant recall remained unchanged. This demonstrates why classification thresholds should be evaluated using multiple metrics rather than accuracy alone.

### Tools Used
Python, Google Colab, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.

*Disclaimer: This project is for educational purposes only and is not intended for medical diagnosis.*

## Task 3: Pipelines and Cross-Validation

### Objective
Build a leak-free machine-learning pipeline and validate it using K-fold cross-validation.

### Work Completed
- Split the data into training and test sets before model evaluation.
- Used a `ColumnTransformer` to handle numerical and categorical features.
- Applied median imputation and standard scaling to numerical features.
- Applied most-frequent imputation and one-hot encoding to categorical features.
- Combined preprocessing and Linear Regression in a single scikit-learn Pipeline.
- Performed 5-fold cross-validation on training data only.
- Evaluated the final pipeline on a separate test set.

### Results
- Mean CV R²: 0.7681
- CV R² standard deviation: 0.0836
- Test MAE: $18,359.52
- Test RMSE: $29,447.16
- Test R²: 0.8869

### Tools Used
Python, Google Colab, Pandas, NumPy, and Scikit-learn.

### Conclusion
The project demonstrates pipeline-based preprocessing, K-fold cross-validation, and separation of training and test data to prevent leakage.
