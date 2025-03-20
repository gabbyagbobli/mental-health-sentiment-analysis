# 🧠 Mental Health Sentiment Analysis

## 📌 Project Overview  
This project performs **sentiment analysis** on mental health-related text data using **Natural Language Processing (NLP) and Machine Learning**. It preprocesses text data, extracts features, and applies multiple classification models to predict sentiment categories.

## 📂 Dataset  
The dataset is stored in **"Combined Data.csv"** and consists of mental health-related text statements labeled with sentiment classes.

## 🛠️ Features & Methodology  
1. **Text Preprocessing**  
   - Convert text to lowercase  
   - Remove punctuation, numbers, and stopwords  
   - Apply **stemming** (reducing words to their root forms)  
   - Convert to a **Document-Term Matrix (DTM)**  

2. **Machine Learning Models**  
   The dataset is trained and evaluated using the following models:
   - **Naïve Bayes (NB)**  
   - **Decision Tree (rpart)**  
   - **Random Forest (RF)**  
   - **Gradient Boosting (GBM)**  
   - **XGBoost**  
   - **Multinomial Logistic Regression**  
   - **Principal Component Analysis (PCA) for Dimensionality Reduction**  
   - **Stacked Ensemble Model (SuperLearner)**  

3. **Model Evaluation**  
   - Models are evaluated using **Confusion Matrix, Accuracy, Precision, Recall, and F1-score**.  
   - Hyperparameter tuning is applied to improve model performance.  

## 🔧 Installation & Setup  
### **1. Clone the repository**  
```bash
git clone https://github.com/gabbyagbobli/mental-health-sentiment-analysis.git
cd mental-health-sentiment-analysis

install.packages(c("tm", "gbm", "SnowballC", "e1071", "caret", "tidyverse", "rpart", 
                   "randomForest", "xgboost", "SuperLearner", "nnet"))

source("mental-health-sentiment-analysis.R")


