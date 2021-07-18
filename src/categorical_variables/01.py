from collections import Counter
from os import getcwd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

### Get data & separate train/test data
file_path = getcwd() + '/src/categorical_variables/data/users.csv'
df = pd.read_csv(file_path)

x_df = df[['home', 'busca', 'logado']]
y_df = df['comprou']

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

total_0s = len(y[y == 'sim'])
total_1s = len(y[y == 'nao'])

print('Total 0s:', total_0s)
print('Total 1s:', total_1s)
print('Number if most frequent value:', max(Counter(y).values()))

### Get predictions & hit rate
def fit_and_predict(model, x_train, y_train, x_test, y_test):
  model.fit(x_train, y_train)

  predictions = model.predict(x_test)
  hits = (predictions == y_test)

  total_hits = sum(hits)
  total_elements = len(x_test)

  hit_rate = 100 * total_hits / total_elements

  print(type(model).__name__, 'hit rate:', hit_rate)

  return hit_rate

model = MultinomialNB()
multinomial_hit_rate = fit_and_predict(model, x_train, y_train, x_test, y_test)

model = AdaBoostClassifier()
ada_boost_hit_rate = fit_and_predict(model, x_train, y_train, x_test, y_test)

### Get most frequent Y (dummy strategy)
dummy_model = DummyClassifier(strategy='most_frequent')
dummy_model.fit(x_train, y_train)
dummy_hit_rate = dummy_model.score(x_test, y_test) * 100

print('DummyClassifier hit hate: %f' % dummy_hit_rate)
