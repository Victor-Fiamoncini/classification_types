import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from os import getcwd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

### Getting data & converting emails to vectors
language = "portuguese"
file_path = getcwd() + '/src/text_processing/data/emails.csv'
classifications = pd.read_csv(file_path, encoding='utf-8')

raw_emails = classifications['email']
lowercase_emails = raw_emails.str.lower()
tokenized_emails = [
  word_tokenize(email, language=language)
  for email in lowercase_emails
]

words = set()
stopwords = nltk.corpus.stopwords.words(language)
stemmer = nltk.stem.RSLPStemmer()

# Removing stopwords and stem & update a word dict/set
for word_list in tokenized_emails:
  non_stopwords_without_suffix = [
    stemmer.stem(word)
    for word in word_list if word not in stopwords and len(word) > 2
  ]
  words.update(non_stopwords_without_suffix)

words_total = len(words)
words_tuples = zip(words, range(words_total))
translator = {word:index for word, index in words_tuples}

def text_to_vector(text, translator):
  vector_phrase = [0] * len(translator)

  for word in text:
    if len(word) > 0:
      word_without_suffix = stemmer.stem(word)

      if word_without_suffix in translator:
        word_occurrence_index = translator[word_without_suffix]
        vector_phrase[word_occurrence_index] += 1

  return vector_phrase

text_vectors = [text_to_vector(email, translator) for email in tokenized_emails]

### Train, Test & Validate data
def fit_and_predict(model, x_train, y_train):
  k = 10
  predictions = cross_val_score(model, x_train, y_train, cv=k)
  score_mean = np.mean(predictions)

  print(type(model).__name__, 'score mean:', score_mean)

  return score_mean

x = np.array(text_vectors)
y = np.array(classifications['classificacao'].tolist())

train_percentage = 0.8
train_size = int(train_percentage * len(y))
validation_size = len(y) - train_size

x_train = x[0:train_size]
y_train = y[0:train_size]

x_validation = x[train_size:]
y_validation = y[train_size:]

results = {}

# K-Fold strategy (with MultinomialNB)
multinomial_model = MultinomialNB()
multinomial_hit_rate = fit_and_predict(multinomial_model, x_train, y_train)
results[multinomial_hit_rate] = multinomial_model

# K-Fold strategy (with AdaBoostClassifier)
ada_boost_model = AdaBoostClassifier()
ada_boost_hit_rate = fit_and_predict(ada_boost_model, x_train, y_train)
results[ada_boost_hit_rate] = ada_boost_model

# K-Fold strategy (with OneVsRestClassifier & LinearSVC)
one_vs_rest_linear_svc_model = OneVsRestClassifier(LinearSVC(random_state=0, max_iter=10000))
one_vs_rest_linear_svc_score_mean = fit_and_predict(one_vs_rest_linear_svc_model, x_train, y_train)
results[one_vs_rest_linear_svc_score_mean] = one_vs_rest_linear_svc_model

# K-Fold strategy (with OneVsOneClassifier & LinearSVC)
one_vs_one_linear_svc_model = OneVsOneClassifier(LinearSVC(random_state=0, max_iter=10000))
one_vs_one_linear_svc_score_mean = fit_and_predict(one_vs_one_linear_svc_model, x_train, y_train)
results[one_vs_one_linear_svc_score_mean] = one_vs_one_linear_svc_model

greater_score_mean = max(results)
winner_model = results[greater_score_mean]

winner_model.fit(x_train, y_train)
winner_predictions = winner_model.predict(x_validation)
hits = (winner_predictions == y_validation)
total_hits = sum(hits)
total_elements = len(x_validation)
score_mean = 100 * total_hits / total_elements

print('Winner', type(winner_model).__name__, 'score mean:', score_mean)

### Get most frequent Y (train, test with dummy strategy)
dummy_model = DummyClassifier(strategy='most_frequent')
dummy_model.fit(x_validation, y_validation)
dummy_hit_rate = dummy_model.score(x_validation, y_validation) * 100

print('DummyClassifier hit hate: %f' % dummy_hit_rate)
