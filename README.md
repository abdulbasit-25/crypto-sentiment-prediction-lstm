# Crypto Sentiment Prediction using NLP & LSTM

## 📌 Project Overview
This project focuses on predicting cryptocurrency market sentiment using **Natural Language Processing (NLP)** and **Machine Learning**, with a primary emphasis on **LSTM (Long Short-Term Memory)** neural networks. The goal is to analyze crypto-related textual data and classify sentiment as **Positive, Neutral, or Negative**, helping identify early market signals driven by public opinion.

---

## Academic Context
- **University:** Shaheed Zulfikar Ali Bhutto Institute of Science & Technology (SZABIST), Islamabad  
- **Department:** Robotics & Artificial Intelligence  
- **Course:** Lab DSA – BS (AI)  
- **Section:** BS (AI) – 3B  

---

## Authors
- **Abdul Basit**  
- **Registration Numbers:** 24108150  

---

## 📊 Problem Motivation
Cryptocurrency markets are highly volatile and influenced more by **public sentiment, media narratives, and investor psychology** than traditional economic indicators. Rapid emotional reactions such as fear and greed often cause sudden price fluctuations.

This project aims to build an AI-based sentiment analysis system that can:
- Quantify market emotions
- Detect early trend shifts
- Support data-driven crypto market insights

---

## Project Objectives
- **Sentiment Classification:** Categorize crypto-related text into Positive, Neutral, or Negative  
- **NLP + ML Integration:** Apply NLP techniques with machine learning and deep learning models  
- **Trend Analysis:** Correlate sentiment trends with cryptocurrency market behavior  

---

## 📁 Dataset
- **Source:** Kaggle – Cryptocurrency Prediction (AI) Dataset  
- **Link:** https://www.kaggle.com/datasets/emirhanai/cryptocurrency-prediction-artificialintelligence  
- **Size:** ~50,000 text entries  
- **Sentiment Labels:** Positive | Neutral | Negative  

### Data Cleaning Steps
- Removed URLs and emojis  
- Eliminated duplicate entries  
- Normalized and tokenized text  

---

##  Technologies & Libraries
- **Programming Language:** Python 3.9.8  
- **Libraries:**  
  - TensorFlow  
  - Keras  
  - NumPy  
  - Pandas  
  - Matplotlib  
  - Scikit-learn  

---

## Methodology
### 1. Data Preprocessing
- Text normalization  
- Tokenization and lemmatization  

### 2. Feature Engineering
- TF-IDF  
- Word embeddings (Word2Vec)

### 3. Model Training
Models evaluated:
- Logistic Regression  
- Support Vector Machine (SVM)  
- Long Short-Term Memory (LSTM)

---

## 📈 Results
| Model | Accuracy |
|------|----------|
| Logistic Regression | 81% |
| SVM | 85% |
| **LSTM (Final Model)** | **91%** |

The LSTM model demonstrated superior performance due to its ability to capture contextual and sequential patterns in text data.

---

## Future Enhancements
- Implement transformer-based models (BERT, RoBERTa)  
- Integrate real-time data from Twitter and Reddit APIs  
- Develop a web dashboard for live sentiment visualization  
- Deploy as a real-time market sentiment monitoring system  

---

## 📌 Conclusion
This project demonstrates how **Artificial Intelligence can transform unstructured textual data into actionable financial insights**. With 91% accuracy, the sentiment prediction model effectively captures early indicators of market behavior driven by public sentiment.

---

## 📚 References
1. Bulut, E. (2021). *Cryptocurrency Prediction with Artificial Intelligence (LSTM Neural Networks)* – Kaggle  
2. VirtualBacik – YouTube: *7 Years of Crypto Trading Knowledge in 50 Minutes*  
3. Wikipedia – Cryptocurrency & Sentiment Analysis  
4. Basic Principles of Economics – CA / CAF Level  

---

## ⚠️ Disclaimer
This project is for **academic and research purposes only** and does not constitute financial or investment advice.
