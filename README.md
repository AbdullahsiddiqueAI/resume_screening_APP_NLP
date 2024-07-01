# Resume Screning APP

This project involves classifying resumes into various categories using a machine learning model. The model is trained on a dataset of resumes, preprocessed to remove noise, and then vectorized using TF-IDF. A K-Nearest Neighbors (KNN) algorithm wrapped in a OneVsRestClassifier is used for classification.

# Dataset
The dataset used is UpdatedResumeDataSet.csv, which contains resumes and their corresponding categories.
https://www.kaggle.com/datasets/jillanisofttech/updated-resume-dataset

# Data Preprocessing
# Exploring Categories:

Count and visualize the distribution of categories.
Create count and pie charts for better understanding of category distribution.
# Cleaning Data:

Remove URLs, hashtags, mentions, special characters, and punctuations from the resumes.
Convert resumes to a clean text format using a custom cleaning function.
Encoding Categories:

Convert categorical values of the target variable into numerical form using LabelEncoder.
Model Building
# TF-IDF Vectorization:

Convert the cleaned resumes into TF-IDF vectors to represent the text data numerically.
# Splitting Data:

Split the dataset into training and testing sets.
# Training the Model:

Train a K-Nearest Neighbors (KNN) classifier using OneVsRest strategy to handle multi-class classification.

# Model Evaluation
Evaluate the model's performance using accuracy score on the test data.
Inference
Build a predictive system that takes a resume as input, preprocesses it, and predicts the category using the trained model.

# Saving the Model
Save the trained model and TF-IDF vectorizer using pickle for future use.