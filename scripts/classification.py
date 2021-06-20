### Build a multi-label model that’s capable of detecting 5 different topics of Reuters newswire
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize, PorterStemmer
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.linear_model import LogisticRegression
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.adapt import MLkNN
from scipy.sparse import lil_matrix

reuters = pd.read_csv("../data/reuters.csv", encoding="utf-8")

cached_stopwords = stopwords.words("english")
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def tokenize(text):
    words = map(lambda word: word, word_tokenize(text))
    # filter out stopwords
    words = [word for word in words if word not in cached_stopwords]
    # stemming
    # tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))
    # lemmatization
    tokens = (list(map(lambda token: lemmatizer.lemmatize(token), words)))
    # remove numbers
    p = re.compile('[a-zA-Z]+')
    filtered_tokens = list(filter(lambda token: p.match(token), tokens))
    return ' '.join(filtered_tokens)



# split data into training and test sets
train = reuters[reuters.train_test == 'TRAIN']
test = reuters[reuters.train_test == 'TEST']
train_text = train['body'].values
test_text = test['body'].values


## Classifiers Training and testing

# The cosine similarity between two vectors is their dot product
vectorizer = TfidfVectorizer(tokenizer=tokenize, strip_accents='unicode', analyzer='word',  norm='l2')
# Learn vocabulary and idf from training set.
vectorizer.fit(train_text)
vectorizer.fit(test_text)
# Convert to a matrix of TF-IDF features
x_train = vectorizer.transform(train_text)
# multilabel binary data
y_train = train.drop(labels=['new_id', 'train_test', 'foc_topics', 'body', 'multi_topics'], axis=1)
x_test = vectorizer.transform(test_text)
y_test = test.drop(labels=['new_id', 'train_test', 'foc_topics', 'body', 'multi_topics'], axis=1)

## 3 Multi-label classification methods

# Binary Relevance
# initialize Binary Relevance multi-label classifier with an SVM classifier
br_classifier = BinaryRelevance(
    classifier=SVC(),
    require_dense=[False, True]
)
br_classifier.fit(x_train, y_train)

# The prediction output is the union of all per label classifiers
br_predictions = br_classifier.predict(x_test)
print("F1 score = ",f1_score(y_test,br_predictions, average="micro"))
# Compute fraction of labels that are incorrectly predicted
print("Hamming loss = ", hamming_loss(y_test, br_predictions))

# Label Powerset
# initialize LabelPowerset multi-label classifier with LogisticRegression
lp_classifier = LabelPowerset(LogisticRegression())
lp_classifier.fit(x_train, y_train)

lp_predictions = lp_classifier.predict(x_test)
print("F1 score = ",f1_score(y_test,lp_predictions, average="micro"))
print("Hamming loss = ", hamming_loss(y_test, lp_predictions))

# MLkNN
ml_classifier = MLkNN(k=10)
# to prevent errors when handling sparse matrices.
x_train = lil_matrix(x_train).toarray()
y_train = lil_matrix(y_train).toarray()
x_test = lil_matrix(x_test).toarray()
ml_classifier.fit(x_train, y_train)

# predict
ml_predictions = ml_classifier.predict(x_test)
print("F1 score = ",f1_score(y_test,ml_predictions, average="micro"))
print("Hamming loss = ", hamming_loss(y_test, ml_predictions))


