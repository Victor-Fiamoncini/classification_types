from os import getcwd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
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
test_size = int(0.1 * len(y))
validation_size = len(y) - train_size - test_size

x_train = x[0:train_size]
y_train = y[0:train_size]

x_test = x[train_size:train_size + test_size]
y_test = y[train_size:train_size + test_size]

x_validation = x[train_size + test_size:]
y_validation = y[train_size + test_size:]

### Get most frequent Y (train, test with dummy strategy)
dummy_model = DummyClassifier(strategy='most_frequent')
dummy_model.fit(x_train, y_train)
dummy_hit_rate = dummy_model.score(x_test, y_test) * 100

print('DummyClassifier hit hate: %f' % dummy_hit_rate)

### Get predictions & hit rate (train, test, validate)
def fit_and_predict(model):
  model.fit(x_train, y_train)

  predictions = model.predict(x_test)
  hits = (predictions == y_test)
  total_hits = sum(hits)
  total_elements = len(x_test)
  hit_rate = 100 * total_hits / total_elements

  print(type(model).__name__, 'hit rate:', hit_rate)

  return hit_rate

results = {}

# MultinomialNB
multinomial_model = MultinomialNB()
multinomial_hit_rate = fit_and_predict(multinomial_model)
results[multinomial_hit_rate] = multinomial_model

# AdaBoost
ada_boost_model = AdaBoostClassifier()
ada_boost_hit_rate = fit_and_predict(ada_boost_model)
results[ada_boost_hit_rate] = ada_boost_model

# One vs Rest strategy (with LinearSVC)
one_vs_rest_linear_svc_model = OneVsRestClassifier(
  LinearSVC(random_state=0, max_iter=10000)
)
one_vs_rest_linear_svc_hit_rate = fit_and_predict(one_vs_rest_linear_svc_model)
results[one_vs_rest_linear_svc_hit_rate] = one_vs_rest_linear_svc_model

# One vs One strategy (with LinearSVC)
one_vs_one_linear_svc_model = OneVsOneClassifier(
  LinearSVC(random_state=0, max_iter=10000)
)
one_vs_one_linear_svc_hit_rate = fit_and_predict(one_vs_one_linear_svc_model)
results[one_vs_one_linear_svc_hit_rate] = one_vs_one_linear_svc_model

greater_hit_rate = max(results)
winner_model = results[greater_hit_rate]

winner_predictions = winner_model.predict(x_validation)
hits = (winner_predictions == y_validation)
total_hits = sum(hits)
total_elements = len(x_validation)
hit_rate = 100 * total_hits / total_elements

print('Winner', type(winner_model).__name__, 'hit rate:', hit_rate)

