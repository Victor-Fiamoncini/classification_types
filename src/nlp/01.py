from os import getcwd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

### Getting data, modeling & separating into train/test data
file_path = getcwd() + '/src/nlp/data/imdb-reviews-pt-br.csv'
df = pd.read_csv(file_path)

classification = df['sentiment'].replace(['neg', 'pos'], [0, 1])
df['classification'] = classification


# sparse_matrix = pd.DataFrame.sparse.from_spmatrix(
#   bag_of_words,
#   columns=vectorize.get_feature_names()
# )

### Predict with LogisticRegression
def classify_text(data_frame, x_column, y_column):
  SEED = 42

  vectorize = CountVectorizer(lowercase=False, max_features=50)
  bag_of_words = vectorize.fit_transform(data_frame[x_column])

  x_train, x_test, y_train, y_test = train_test_split(
    bag_of_words,
    data_frame[y_column],
    random_state=SEED
  )

  logistic_regression_model = LogisticRegression()
  logistic_regression_model.fit(x_train, y_train)
  accuracy = logistic_regression_model.score(x_test, y_test)

  print(accuracy)
