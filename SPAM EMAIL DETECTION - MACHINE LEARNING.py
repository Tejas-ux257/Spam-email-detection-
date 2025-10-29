# SPAM EMAIL DETECTION - MACHINE LEARNING MODEL


"""
This notebook implements a spam email classifier using:
- Scikit-learn for machine learning
- Multiple classification algorithms
- Text preprocessing with TF-IDF
- Model evaluation and comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

print("=" * 60)
print("SPAM EMAIL DETECTION - MACHINE LEARNING MODEL")
print("CODTECH Internship Project")
print("=" * 60)

# ============================================================
# PART 2: CREATE SAMPLE DATASET
# ============================================================

print("\n[1] Creating Sample Dataset...")

# Sample spam and ham (non-spam) emails
spam_emails = [
    "Congratulations! You've won $1000000. Click here to claim now!",
    "FREE MONEY! Act now and get rich quick!",
    "You have won a lottery! Send your bank details immediately.",
    "URGENT: Your account will be closed. Verify now!",
    "Get cheap medications online. No prescription needed!",
    "Make money fast! Work from home opportunity!",
    "Click here for a FREE iPhone! Limited offer!",
    "You've been selected for a special prize. Claim now!",
    "Lose weight fast with this miracle pill!",
    "Earn $5000 per week working from home!",
    "WINNER! You've won a vacation package to Hawaii!",
    "Increase your income overnight! No effort required!",
    "Buy Viagra online at lowest prices!",
    "Your loan has been approved! Click here!",
    "Get a degree online in just 2 weeks!",
    "Amazing deal! 90% discount on luxury watches!",
    "Free credit card with no fees! Apply now!",
    "You are a lucky winner! Collect your prize!",
    "Hot singles in your area want to meet you!",
    "Refinance your home at the lowest rates!"
]

ham_emails = [
    "Hey, are we still meeting for lunch tomorrow?",
    "The project deadline has been extended to next Friday.",
    "Can you send me the report when you get a chance?",
    "Thanks for your help with the presentation yesterday.",
    "Meeting scheduled for 3 PM in conference room B.",
    "Your Amazon order has been shipped and will arrive tomorrow.",
    "Reminder: Doctor's appointment on Thursday at 10 AM.",
    "The team dinner is planned for next Saturday evening.",
    "Please review the attached document and provide feedback.",
    "Your subscription renewal is coming up next month.",
    "Great job on the client presentation today!",
    "Can you pick up milk on your way home?",
    "The conference call notes are attached for your review.",
    "Your flight confirmation for next week is attached.",
    "Happy birthday! Hope you have a wonderful day!",
    "The system maintenance is scheduled for tonight at 11 PM.",
    "Your reservation at the restaurant has been confirmed.",
    "Please find the quarterly report attached.",
    "Looking forward to working with you on this project.",
    "Your package has been delivered to the front desk."
]

# Create DataFrame
data = pd.DataFrame({
    'email': spam_emails + ham_emails,
    'label': ['spam'] * len(spam_emails) + ['ham'] * len(ham_emails)
})

# Shuffle the data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"✓ Dataset created with {len(data)} emails")
print(f"  - Spam emails: {len(spam_emails)}")
print(f"  - Ham emails: {len(ham_emails)}")

# Display sample data
print("\nSample Emails:")
print(data.head(10))

# ============================================================
# PART 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

print("\n" + "=" * 60)
print("[2] Exploratory Data Analysis")
print("=" * 60)

# Label distribution
print("\nLabel Distribution:")
print(data['label'].value_counts())

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Pie chart
data['label'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=axes[0], colors=['#ff9999', '#66b3ff'])
axes[0].set_title('Spam vs Ham Distribution')
axes[0].set_ylabel('')

# Bar chart
data['label'].value_counts().plot(kind='bar', ax=axes[1], color=['#ff9999', '#66b3ff'])
axes[1].set_title('Email Count by Category')
axes[1].set_xlabel('Category')
axes[1].set_ylabel('Count')
axes[1].set_xticklabels(['Spam', 'Ham'], rotation=0)

plt.tight_layout()
plt.savefig('email_distribution.png', dpi=300, bbox_inches='tight')
print("\n✓ Distribution plot saved as 'email_distribution.png'")

# Text length analysis
data['length'] = data['email'].apply(len)

print("\nEmail Length Statistics:")
print(data.groupby('label')['length'].describe())

# ============================================================
# PART 4: DATA PREPROCESSING
# ============================================================

print("\n" + "=" * 60)
print("[3] Data Preprocessing")
print("=" * 60)

# Convert labels to binary (0 for ham, 1 for spam)
data['label_encoded'] = data['label'].map({'ham': 0, 'spam': 1})

# Split features and target
X = data['email']
y = data['label_encoded']

# Split into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Data split completed:")
print(f"  - Training samples: {len(X_train)}")
print(f"  - Testing samples: {len(X_test)}")

# Text Vectorization using TF-IDF
print("\n✓ Applying TF-IDF Vectorization...")
tfidf = TfidfVectorizer(max_features=100, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"  - Feature dimensions: {X_train_tfidf.shape[1]}")
print(f"  - Training matrix shape: {X_train_tfidf.shape}")
print(f"  - Testing matrix shape: {X_test_tfidf.shape}")

# ============================================================
# PART 5: MODEL IMPLEMENTATION
# ============================================================

print("\n" + "=" * 60)
print("[4] Model Training and Evaluation")
print("=" * 60)

# Dictionary to store model results
model_results = {}

# Function to train and evaluate models
def train_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """Train and evaluate a machine learning model"""
    
    print(f"\n{'─' * 50}")
    print(f"Training: {model_name}")
    print(f"{'─' * 50}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    cv_mean = cv_scores.mean()
    
    # Store results
    model_results[model_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cv_score': cv_mean,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'model': model
    }
    
    # Print results
    print(f"\n✓ Results:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  CV Score:  {cv_mean:.4f} (±{cv_scores.std():.4f})")
    
    return model, y_pred

# ============================================================
# MODEL 1: NAIVE BAYES
# ============================================================

nb_model = MultinomialNB()
nb_model, nb_pred = train_evaluate_model(
    nb_model, "Naive Bayes", X_train_tfidf, X_test_tfidf, y_train, y_test
)

# ============================================================
# MODEL 2: LOGISTIC REGRESSION
# ============================================================

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model, lr_pred = train_evaluate_model(
    lr_model, "Logistic Regression", X_train_tfidf, X_test_tfidf, y_train, y_test
)

# ============================================================
# MODEL 3: SUPPORT VECTOR MACHINE
# ============================================================

svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model, svm_pred = train_evaluate_model(
    svm_model, "Support Vector Machine", X_train_tfidf, X_test_tfidf, y_train, y_test
)

# ============================================================
# MODEL 4: RANDOM FOREST
# ============================================================

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model, rf_pred = train_evaluate_model(
    rf_model, "Random Forest", X_train_tfidf, X_test_tfidf, y_train, y_test
)

# ============================================================
# PART 6: MODEL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("[5] Model Comparison")
print("=" * 60)

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Model': list(model_results.keys()),
    'Accuracy': [results['accuracy'] for results in model_results.values()],
    'Precision': [results['precision'] for results in model_results.values()],
    'Recall': [results['recall'] for results in model_results.values()],
    'F1-Score': [results['f1'] for results in model_results.values()],
    'CV Score': [results['cv_score'] for results in model_results.values()]
})

print("\nModel Performance Comparison:")
print(comparison_df.to_string(index=False))

# Find best model
best_model_name = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Accuracy: {comparison_df['Accuracy'].max():.4f}")

# Visualize model comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

for idx, (ax, metric, color) in enumerate(zip(axes.flat, metrics, colors)):
    comparison_df.plot(x='Model', y=metric, kind='bar', ax=ax, color=color, legend=False)
    ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel(metric)
    ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
    ax.set_ylim([0, 1.1])
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Comparison plot saved as 'model_comparison.png'")

# ============================================================
# PART 7: CONFUSION MATRIX VISUALIZATION
# ============================================================

print("\n" + "=" * 60)
print("[6] Confusion Matrix Analysis")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, (model_name, ax) in enumerate(zip(model_results.keys(), axes.flat)):
    y_pred = model_results[model_name]['y_pred']
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    ax.set_title(f'{model_name}\nConfusion Matrix', fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
print("\n✓ Confusion matrices saved as 'confusion_matrices.png'")

# ============================================================
# PART 8: CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("[7] Detailed Classification Reports")
print("=" * 60)

for model_name in model_results.keys():
    y_pred = model_results[model_name]['y_pred']
    print(f"\n{model_name}:")
    print("─" * 50)
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# ============================================================
# PART 9: TEST WITH NEW EMAILS
# ============================================================

print("\n" + "=" * 60)
print("[8] Testing with New Emails")
print("=" * 60)

# New test emails
new_emails = [
    "Congratulations! You won a free ticket to Bahamas!",
    "Hey, can you review my code when you have time?",
    "URGENT: Verify your account now or it will be deleted!",
    "Meeting rescheduled to Thursday at 2 PM",
    "Get rich quick! Invest now and earn millions!"
]

# Vectorize new emails
new_emails_tfidf = tfidf.transform(new_emails)

# Predict with best model
best_model = model_results[best_model_name]['model']
predictions = best_model.predict(new_emails_tfidf)
prediction_proba = best_model.predict_proba(new_emails_tfidf)

print(f"\nPredictions using {best_model_name}:\n")
for i, (email, pred, proba) in enumerate(zip(new_emails, predictions, prediction_proba), 1):
    label = 'SPAM' if pred == 1 else 'HAM'
    confidence = proba[pred] * 100
    print(f"{i}. Email: {email[:60]}...")
    print(f"   Prediction: {label} (Confidence: {confidence:.2f}%)\n")

# ============================================================
# PART 10: FEATURE IMPORTANCE (FOR BEST MODEL)
# ============================================================

print("\n" + "=" * 60)
print("[9] Feature Importance Analysis")
print("=" * 60)

if best_model_name in ["Logistic Regression", "Support Vector Machine"]:
    # Get feature names
    feature_names = tfidf.get_feature_names_out()
    
    # Get coefficients
    if hasattr(best_model, 'coef_'):
        coefficients = best_model.coef_[0]
        
        # Create DataFrame
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients
        }).sort_values('coefficient', key=abs, ascending=False)
        
        print("\nTop 10 Spam Indicators:")
        print(feature_importance.head(10).to_string(index=False))
        
        print("\nTop 10 Ham Indicators:")
        print(feature_importance.tail(10).to_string(index=False))
        
        # Visualize top features
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Top spam features
        top_spam = feature_importance.head(10)
        ax1.barh(top_spam['feature'], top_spam['coefficient'], color='#e74c3c')
        ax1.set_xlabel('Coefficient')
        ax1.set_title('Top 10 Spam Indicators', fontweight='bold')
        ax1.invert_yaxis()
        
        # Top ham features
        top_ham = feature_importance.tail(10)
        ax2.barh(top_ham['feature'], top_ham['coefficient'], color='#2ecc71')
        ax2.set_xlabel('Coefficient')
        ax2.set_title('Top 10 Ham Indicators', fontweight='bold')
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("\n✓ Feature importance plot saved as 'feature_importance.png'")

# ============================================================
# PART 11: FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)

print(f"""
✓ Dataset: {len(data)} emails ({len(spam_emails)} spam, {len(ham_emails)} ham)
✓ Training samples: {len(X_train)}
✓ Testing samples: {len(X_test)}
✓ Feature extraction: TF-IDF Vectorization
✓ Models trained: 4 (Naive Bayes, Logistic Regression, SVM, Random Forest)
✓ Best model: {best_model_name}
✓ Best accuracy: {comparison_df['Accuracy'].max():.4f} ({comparison_df['Accuracy'].max()*100:.2f}%)

Files Generated:
  1. email_distribution.png - Dataset visualization
  2. model_comparison.png - Model performance comparison
  3. confusion_matrices.png - Confusion matrices for all models
  4. feature_importance.png - Important features for classification

CODTECH Internship - Machine Learning Model Implementation Completed!
""")

print("=" * 60)