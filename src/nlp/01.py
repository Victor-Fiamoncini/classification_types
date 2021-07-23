from os import getcwd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

### Get data & separate train/test data
file_path = getcwd() + '/src/nlp/data/imdb-reviews-pt-br.csv'
df = pd.read_csv(file_path)

seed = 42
x_train, x_test, y_train, y_test = train_test_split(
  df['text_pt'],
  df['sentiment'],
  random_state=seed
)

logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(x_train, y_train)
accuracy = logistic_regression_model.score(x_test, y_test)

print(accuracy)
