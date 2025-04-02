# Binary Rainfall Prediction using Random Forest

---

## Project Description

This notebook presents a full pipeline for **binary classification of rainfall** based on weather conditions. It uses data from the Kaggle competition **"Playground Series S5E3"** and applies a **Random Forest Classifier** to determine whether it will rain (`1`) or not (`0`).

## Features Used
- Pressure
- Max / Min / Mean Temperature
- Dew Point
- Humidity
- Cloud Cover
- Sunshine Duration
- Wind Speed

---

## Workflow

### 📥 Data Loading
```python
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
```

### 🔎 Exploratory Data Analysis
- Used `.describe()` to inspect distributions
- Verified missing values and handled one (`winddirection`) in test set
```python
test["winddirection"] = test["winddirection"].fillna(0)
```

### 🧹 Data Preprocessing
- One-hot encoded categorical features (though there are few)
- Removed irrelevant columns (`id`, `day`, `winddirection`)
```python
train = train.drop(["id", "day", "winddirection"], axis=1)
```

### 📊 Feature Correlation Visualization
```python
sns.heatmap(train.corr(), annot=False)
```

---

## Modeling

### ✅ Model: Random Forest
```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
```

### 🔍 Evaluation
```python
y_pred = rf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
```
Achieved baseline accuracy: **84.47%**

### 🌲 Decision Tree Visualization
Visualized first 3 trees from the forest with `export_graphviz` and `graphviz`.

---

## 🔧 Hyperparameter Tuning
Used `RandomizedSearchCV` with:
```python
param_dist = {'n_estimators': randint(50,500), 'max_depth': randint(1,20)}
```
Final model achieved **86.30% accuracy** on validation.
```python
Best hyperparameters: {'max_depth': 4, 'n_estimators': 348}
```

### 🧠 Confusion Matrix
```python
cm = confusion_matrix(y_val, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
```

---

## ✅ Final Submission
```python
features = [
    'pressure', 'maxtemp', 'temparature', 'mintemp',
    'dewpoint', 'humidity', 'cloud','sunshine','windspeed']
predicted = best_rf.predict(test[features])
```
Output:
```python
submission = pd.DataFrame({"id": test.id, "rainfall": predicted})
submission.to_csv("submission.csv", index=False)
```
✅ "Your submission was successfully saved!"

---

## Summary

- Built a rainfall prediction model with **86%+ accuracy**
- Visualized model internals with decision trees
- Applied EDA, preprocessing, tuning, and evaluation

---

## Improvements
- Use gradient boosting (e.g. XGBoost, LightGBM)
- Add time-based features like rolling averages
- Feature importance and SHAP analysis

---

