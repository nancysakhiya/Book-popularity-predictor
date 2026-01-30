# 📚 What Makes a Good Book?
## Book Popularity Prediction using Machine Learning

### Project Overview
This project aims to identify the factors that contribute to a book’s popularity by building a machine learning model that predicts whether a book is likely to be popular. The model leverages **both textual data** (book titles and descriptions) and **numerical metadata** (ratings, reviews, and page counts), simulating a real-world online bookstore scenario.

The project demonstrates an end-to-end machine learning workflow, including data cleaning, feature engineering, natural language processing (NLP), model building, evaluation, and visualization.

### Problem Statement
Online bookstores and recommendation platforms need to predict which books are likely to become popular in order to:
- Improve recommendation systems  
- Optimize promotions and rankings  
- Enhance user engagement  

This project frames popularity prediction as a **supervised binary classification problem**.

### Learning Type
- **Supervised Learning**
- **Binary Classification**
- **Hybrid ML (NLP + Tabular Data)**

### Dataset
The dataset is sourced from the **Goodreads Books dataset**, containing:
- Book titles and descriptions
- Average ratings
- Ratings count and review count
- Number of pages
Dataset link: https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks


A custom **popularity score** was engineered using ratings and engagement metrics.

### Feature Engineering
- **Text Features**
  - TF-IDF vectorization applied to book descriptions
- **Numerical Features**
  - Average rating
  - Ratings count (log-transformed)
  - Review count
  - Number of pages
- **Missing Value Handling**
  - Text fields filled with empty strings
  - Numerical features imputed using median values

### Model Pipeline
A clean and reusable machine learning pipeline was built using:
- `ColumnTransformer` for handling mixed data types
- `TF-IDF Vectorizer` for text processing
- `StandardScaler` for numerical normalization
- `Logistic Regression` classifier

This approach prevents data leakage and ensures reproducibility.

### Model Evaluation
The model was evaluated using:
- Accuracy
- Precision, Recall, and F1-score
- Confusion Matrix
- ROC-AUC Curve

Visualizations were created using **Matplotlib** and **Seaborn** to clearly communicate results.

### Key Visual Insights
- Popular books tend to have significantly higher review and rating counts
- Certain keywords and themes appear more frequently in popular book descriptions
- The model effectively distinguishes popular books with a strong ROC-AUC score

### Technologies Used
- Python
- Pandas & NumPy
- Scikit-learn
- Matplotlib & Seaborn
- NLP (TF-IDF, WordCloud)

### How to Run the Project

1. Clone the repository: 
```bash
https://github.com/nancysakhiya/Book-popularity-predictor.git
