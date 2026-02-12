📧 Spam Email Detection Model

This project is a Spam Email Detection machine learning model built using Python and scikit-learn.
The model classifies emails as Spam or Not Spam (Ham) using Naive Bayes algorithms.

🚀 Features

Text preprocessing using CountVectorizer

Model training using:

Multinomial Naive Bayes

Bernoulli Naive Bayes

Dataset split using train_test_split

Simple and efficient spam classification

🛠️ Technologies Used

Python 🐍

scikit-learn

pandas

NumPy

📚 Libraries Used
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

📂 Project Structure
spam-email-detection/
│
├── data/                  # Dataset files
├── spam_detector.ipynb    # Jupyter Notebook
├── spam_detector.py       # Python script (if any)
├── README.md              # Project documentation
└── requirements.txt       # Required libraries

⚙️ How It Works (Simple Explanation)

Load Dataset
Email text and labels (spam / ham)

Text Vectorization
Convert email text into numbers using CountVectorizer

Split Data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


Train Model

MultinomialNB → best for word frequency

BernoulliNB → best for binary features (word present or not)

Predict & Evaluate
The model predicts whether an email is Spam or Not Spam

📊 Models Used
🔹 Multinomial Naive Bayes

Works well with word counts

Commonly used for text classification

🔹 Bernoulli Naive Bayes

Works with binary features

Checks whether a word exists or not

▶️ How to Run the Project

Clone the repository

git clone https://github.com/your-username/spam-email-detection.git


Install dependencies

pip install -r requirements.txt


Run the notebook or script

jupyter notebook


or

python spam_detector.py

✅ Output Example
Email: "Congratulations! You won a free prize"
Prediction: SPAM

🎯 Future Improvements

Use TF-IDF Vectorizer

Try Logistic Regression

Add confusion matrix & accuracy score

Deploy as a web app
