# 📧 SMS Spam Detection (Machine Learning)

A Machine Learning project that classifies SMS messages as Spam or Ham (Not Spam) using Natural Language Processing (NLP) and Naive Bayes algorithms.

# 🚀 Features

Text preprocessing and cleaning

SMS text vectorization using CountVectorizer

Spam classification using Naive Bayes

Model pipeline for simplified workflow

Achieves ~98% accuracy

# 🧠 Model Used

Multinomial Naive Bayes

Bernoulli Naive Bayes

Text is converted into numerical features using:

CountVectorizer
# 📊 Dataset

5570 SMS messages

Two classes:

ham → Normal message

spam → Unwanted message

# ⚙️ Technologies

Python

Pandas

Scikit-learn

NLP (CountVectorizer)

# 📈 Model Accuracy
Model	Accuracy
MultinomialNB	98.02%
BernoulliNB	97.48%
# 📂 Project Structure
spam-detection
```
spam-detection/
│
├── Spam.csv
├── spam_detection.ipynb
└── README.md
```
# ▶️ How to Run

Clone the repository
```
git clone https://github.com/your-username/spam-detection.git
```
