# Mental Health Sentiment Analysis Project

## 📌 Project Overview
This project focuses on analyzing sentiment in mental-health–related text data using Natural Language Processing (NLP) and Machine Learning techniques. The aim is to classify text into sentiment categories and explore correlations between sentiment and mental‑health indicators.

---

## 🎯 Objectives
- Perform text preprocessing (cleaning, tokenization, stopword removal, lemmatization).
- Apply sentiment analysis techniques.
- Build ML models (Logistic Regression, SVM, Random Forest, etc.).
- Evaluate model performance using metrics such as accuracy, precision, recall, F1-score.
- Visualize data insights.

---

## 🗂️ Dataset Description
The dataset contains text posts and sentiment labels. Example columns:
- **text** – The user-generated content.
- **sentiment** – Label such as *positive*, *negative*, or *neutral*.

> *(Note: Your sample file of first 5000 rows is used for demonstration.)*

---

## 🧹 Data Preprocessing Steps
1. Remove URLs, special characters, and numbers.
2. Convert text to lowercase.
3. Remove stopwords using NLTK.
4. Lemmatize words.
5. Vectorize text using TF-IDF.

---

## 🧠 Machine Learning Models
Models implemented:
- Logistic Regression
- Support Vector Machine (SVM)
- Naive Bayes
- Random Forest

Each model is trained and evaluated, and performance metrics are compared.

---

## 📊 Evaluation Metrics
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

Warnings handled:
- `UndefinedMetricWarning` solved using: `precision_score(..., zero_division=1)`

---

## 📈 Visualizations
- Sentiment distribution bar plot
- Word clouds for each sentiment
- Confusion matrices for model comparison

---

## 🛠️ Tools & Libraries
- Python
- Pandas, NumPy
- NLTK
- Scikit-learn
- Matplotlib, Seaborn

---

## 🚀 Key Results
- Best-performing model identified based on F1-score.
- Sentiment trends reveal common emotional patterns in mental‑health posts.

---

## 📥 Future Improvements
- Use deep learning models like LSTM or BERT.
- Expand dataset for better generalization.
- Tune hyperparameters using GridSearchCV.

---

## 🙌 Conclusion
This project successfully analyzes sentiment in mental-health-related text and demonstrates how NLP and ML can be applied to support mental‑wellbeing research.

