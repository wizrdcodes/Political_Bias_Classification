# Classifying Political Bias in News Articles
A PySpark machine learning pipeline that classifies news articles as left-leaning, centre, or right-leaning using NLP on article titles and body text.

# Dataset
Political Bias Analysis Dataset — Kaggle. Each row is a news article labelled with its political leaning.

# Approach
 - Text preprocessing with tokenisation, stop-word removal, and TF-IDF vectorisation via PySpark MLlib
 - Models compared: Logistic Regression and Naïve Bayes
 - Hyperparameter tuning with TrainValidationSplit
 - Evaluation via precision, recall, F1-score, and confusion matrices

# Tools
Python · PySpark · pandas · matplotlib

# Context
Developed as part of the CN7030 Machine Learning on Big Data module, MSc Artificial Intelligence and Data Science, University of East London (2025–26).
