# 📧 Spam Email Detection using Machine Learning

This project builds a **Spam Email Classifier** using various **Machine Learning models** to distinguish between spam and ham (non-spam) emails.  
It was developed as part of the **CODTECH Internship Project**.

---

## 🚀 Features
- Preprocessed text data using **TF-IDF Vectorization**
- Trained multiple models:
  - Naive Bayes  
  - Logistic Regression  
  - Support Vector Machine (SVM)  
  - Random Forest  
- Compared models on Accuracy, Precision, Recall, and F1-score
- Visualized model performance and confusion matrices
- Tested on new unseen emails

---

## 🧠 Technologies Used
- Python
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- TF-IDF Vectorizer for feature extraction

---

## 📊 Results
| Model | Accuracy | Precision | Recall | F1-Score |
|--------|-----------|------------|---------|-----------|
| Naive Bayes | ~0.95 | 0.94 | 0.96 | 0.95 |
| Logistic Regression | ~0.97 | 0.96 | 0.97 | 0.96 |
| SVM | ~0.98 | 0.98 | 0.98 | 0.98 |
| Random Forest | ~0.96 | 0.95 | 0.96 | 0.96 |

🏆 **Best Model:** Support Vector Machine (SVM)

---

## 📂 Files Generated
1. `email_distribution.png` — Dataset visualization  
2. `model_comparison.png` — Model comparison chart  
3. `confusion_matrices.png` — Confusion matrices for models  
4. `feature_importance.png` — Top spam/ham indicators  

## 🧾 requirements.txt
numpy
pandas
matplotlib
seaborn
scikit-learn
