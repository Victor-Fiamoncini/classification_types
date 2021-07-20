from os import getcwd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd

### Get data & separate train/test data
file_path = getcwd() + '/src/multiple_categories/data/customers.csv'
df = pd.read_csv(file_path)

x_df = df[['recencia', 'frequencia', 'semanas_de_inscricao']]
y_df = df['situacao']

x_dummies_df = pd.get_dummies(x_df)
y_dummies_df = y_df

x = x_dummies_df.values
y = y_dummies_df.values

train_size = int(0.8 * len(y))

x_train = x[0:train_size]
y_train = y[0:train_size]

x_validation = x[train_size:]
y_validation = y[train_size:]

### Get predictions & score mean with K-Fold (train, test, validate)
def fit_and_predict(model):
  k = 10
  predictions = cross_val_score(model, x_train, y_train, cv=k)
  score_mean = np.mean(predictions)

  print(type(model).__name__, 'score mean:', score_mean)

results = {}

# K-Fold strategy (with MultinomialNB)
multinomial_model = MultinomialNB()
multinomial_hit_rate = fit_and_predict(multinomial_model)
results[multinomial_hit_rate] = multinomial_model

# K-Fold strategy (with AdaBoostClassifier)
ada_boost_model = AdaBoostClassifier()
ada_boost_hit_rate = fit_and_predict(ada_boost_model)
results[ada_boost_hit_rate] = ada_boost_model

# K-Fold strategy (with OneVsRestClassifier & LinearSVC)
one_vs_rest_linear_svc_model = OneVsRestClassifier(
  LinearSVC(random_state=0, max_iter=10000)
)
one_vs_rest_linear_svc_score_mean = fit_and_predict(one_vs_rest_linear_svc_model)
results[one_vs_rest_linear_svc_score_mean] = one_vs_rest_linear_svc_model

# K-Fold strategy (with OneVsOneClassifier & LinearSVC)
one_vs_one_linear_svc_model = OneVsOneClassifier(
  LinearSVC(random_state=0, max_iter=10000)
)
one_vs_one_linear_svc_score_mean = fit_and_predict(one_vs_one_linear_svc_model)
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
