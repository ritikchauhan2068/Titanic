# 🚢 Titanic Survival Prediction

A machine learning project that explores the Titanic passenger dataset, compares classification algorithms, and provides an interactive web application for predicting passenger survival.

## 📌 Project Overview

The objective is to develop a binary classification model that predicts whether a passenger survived the Titanic disaster based on passenger information.

- **Target variable:** `Survived`
- `0` – Did not survive
- `1` – Survived
- **Application model:** Random Forest Classifier

## ✨ Key Features

- Exploratory Data Analysis (EDA)
- Missing-value and duplicate analysis
- Data preprocessing using Scikit-learn pipelines
- Comparison of four classification algorithms
- Stratified 5-fold cross-validation
- Hyperparameter tuning using GridSearchCV
- Model evaluation and comparison
- Interactive Streamlit prediction application
- Survival probability visualization

## 🧰 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning |
| Streamlit | Web application |
| Pickle | Model serialization |

## 📁 Project Structure

```text
Titanic/
│
├── st.py
├── titanic.ipynb.ipynb
├── titanic_model11.pkl
├── requirements.txt
└── README.md
```

## 📊 Machine Learning Workflow

### 1. Data Inspection

The dataset contains 891 rows and 12 columns.

Performed:
- Data type inspection
- Missing-value analysis
- Duplicate detection
- Descriptive statistical analysis
- Target distribution analysis

### 2. Exploratory Data Analysis

Investigated:
- Survival distribution
- Survival by passenger class
- Survival by gender
- Age distribution
- Fare distribution
- Outlier detection

### 3. Data Preprocessing

Removed unnecessary columns:

- PassengerId
- Name
- Ticket
- Cabin

Selected features:

**Numerical features**
- Age
- Fare
- SibSp
- Parch

**Categorical features**
- Pclass
- Sex
- Embarked

Applied:
- Median imputation
- Most-frequent imputation
- StandardScaler
- OneHotEncoder
- ColumnTransformer
- Pipeline

Used an 80:20 train-test split with stratification and random_state=42.

### 4. Model Training

Compared four machine learning algorithms:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Random Forest

### 5. Cross-Validation

Implemented Stratified 5-Fold Cross-Validation.

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### 6. Hyperparameter Optimization

Used GridSearchCV to optimize:

- Logistic Regression
- KNN
- Decision Tree
- Random Forest

Selected the final candidate based on the highest mean cross-validation F1-score.

### 7. Model Evaluation

Evaluated the selected model using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Classification Report
- ROC Curve

Refer to the executed notebook for actual performance scores.

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/ritikchauhan2068/Titanic.git

cd Titanic
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run st.py
```

## 🖥️ Application Usage

1. Enter passenger information.
2. Select passenger class and gender.
3. Enter age and ticket fare.
4. Select the embarkation port.
5. Enter family information.
6. Click Predict Survival.
7. View the prediction and probability visualization.

## 📥 Input Features

| Feature | Description |
|---|---|
| Pclass | Passenger ticket class |
| Sex | Passenger gender |
| Age | Passenger age |
| Fare | Ticket fare |
| SibSp | Siblings / Spouses aboard |
| Parch | Parents / Children aboard |
| Embarked | Port of embarkation |

## 🔮 Future Improvements

- SHAP-based model explainability
- Feature importance visualization
- Additional classification algorithms
- Automated model testing
- Model deployment
- Improved prediction interface

## 👨‍💻 Author

**Ritik Chauhan**

GitHub: [@ritikchauhan2068](https://github.com/ritikchauhan2068)

Project: [Titanic Survival Prediction](https://github.com/ritikchauhan2068/Titanic)

Project Dashboard checkout: [Live](https://titanic-fbrunjuzvkrjf9rtwmh6rx.streamlit.app/)

---

⭐ If you find this project useful, consider starring the repository.
