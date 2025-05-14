# Lung Cancer Prediction Using Machine Learning

This project focuses on the detection of lung cancer using a dataset consisting of various symptoms and risk factors. The notebook explores preprocessing techniques, model training, evaluation, and insights to build effective prediction systems for lung cancer using supervised machine learning algorithms.

---

## 📁 Table of Contents

1. [Objective](#objective)
2. [Dataset](#dataset)
3. [Dependencies](#dependencies)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Data Preprocessing](#data-preprocessing)
6. [Model Building](#model-building)
7. [Model Evaluation](#model-evaluation)
8. [Conclusion](#conclusion)
9. [Future Work](#future-work)
10. [License](#license)

---

## 🎯 Objective

To build a machine learning model that can predict the presence of lung cancer based on various lifestyle, health, and demographic attributes. The aim is to assist healthcare professionals in identifying high-risk individuals early.

---

## 📊 Dataset

The dataset includes the following features:

- `GENDER`: Male or Female  
- `AGE`: Age of the individual  
- `SMOKING`: Smoker (1) or Non-smoker (0)  
- `YELLOW_FINGERS`: Presence of yellow fingers due to smoking  
- `ANXIETY`, `PEER_PRESSURE`, `CHRONIC DISEASE`, `FATIGUE`, `ALLERGY`, `WHEEZING`, `ALCOHOL CONSUMING`, `COUGHING`, `SHORTNESS OF BREATH`, `SWALLOWING DIFFICULTY`, `CHEST PAIN`: Binary indicators of symptoms  
- `LUNG_CANCER`: Target class (YES or NO)

**Target Variable**: `LUNG_CANCER`

---

## 🧰 Dependencies

Install the following Python packages before running the notebook:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

You can also use the provided `requirements.txt` file (to be created from the notebook environment).

---

## 🔍 Exploratory Data Analysis

### Basic Exploration:

- Shape of dataset
- Info and data types
- Null value check
- Value counts of target label `LUNG_CANCER`

### Visualizations:

- Countplots for categorical features (e.g., Gender, Smoking)
- Pie chart for lung cancer distribution
- Heatmap showing feature correlation

---

## 🔄 Data Preprocessing

### Label Encoding

Categorical columns like `GENDER` and `LUNG_CANCER` were label-encoded using `LabelEncoder()` from `sklearn`.

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['GENDER'] = le.fit_transform(df['GENDER'])
df['LUNG_CANCER'] = le.fit_transform(df['LUNG_CANCER'])
```

### Splitting Dataset

Split into features and target:

```python
X = df.drop(columns='LUNG_CANCER')
y = df['LUNG_CANCER']
```

Train/test split:

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 🤖 Model Building

Trained various classification models:

- Logistic Regression
- Support Vector Classifier
- Decision Tree
- Random Forest
- K-Nearest Neighbors

Each model was trained and tested using scikit-learn’s implementation.

Example:

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

## 📈 Model Evaluation

### Metrics Used:

- Accuracy Score
- Confusion Matrix
- Classification Report

Example:

```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print("Accuracy:", accuracy_score(y_test, predictions))
```

### Summary of Results:

| Model                 | Accuracy |
|----------------------|----------|
| Logistic Regression  | ~91%     |
| SVM (Linear Kernel)  | ~91%     |
| Decision Tree        | ~97%     |
| Random Forest        | ~98%     |
| KNN                  | ~93%     |

*(Note: These are approximations from notebook outputs. Exact scores depend on your dataset state.)*

---

## ✅ Conclusion

- Random Forest and Decision Tree models showed the highest accuracy.
- Features such as Smoking, Yellow Fingers, and Chest Pain had strong correlations with lung cancer diagnosis.
- Machine learning can aid in early lung cancer detection when combined with reliable data.

---

## 🔮 Future Work

- Use cross-validation for robust evaluation
- Apply hyperparameter tuning (e.g., GridSearchCV)
- Explore ensemble stacking techniques
- Integrate model into a web-based application using Streamlit or Flask

---

## 📜 License

This project is licensed under the MIT License. Feel free to modify and use it for educational or commercial purposes.

---

## 📧 Contact

For queries or contributions, please contact [iamnaveen1401@gmail.com]
