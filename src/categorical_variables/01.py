from os import getcwd
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

train_size = int(0.9 * len(y))
test_size = len(y) - train_size

x_train = x[:train_size]
y_train = y[:train_size]

x_test = x[-test_size:]
y_test = y[-test_size:]

### Get predictions & hit rate
model = MultinomialNB()
model.fit(x_train, y_train)

predictions = model.predict(x_test)
differences = predictions - y_test

hits = [d for d in differences if d == 0]
total_hits = len(hits)
total_elements = len(x_test)

hit_rate = 100 * total_hits / total_elements

print(hit_rate, total_elements)
