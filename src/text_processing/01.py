from os import getcwd
import pandas as pd

### Getting data & converting emails to vectors
file_path = getcwd() + '/src/text_processing/data/emails.csv'
classifications = pd.read_csv(file_path)

raw_emails = classifications['email']
splitted_emails = raw_emails.str.lower().str.split(' ')

words = set()
for word_list in splitted_emails:
  words.update(word_list)

words_total = len(words)
words_tuples = zip(words, range(words_total))
translator = {word:index for word, index in words_tuples}

def text_to_vector(text, translator):
  vector_phrase = [0] * len(translator)

  for word in text:
    if word in translator:
      word_occurrence_index = translator[word]
      vector_phrase[word_occurrence_index] += 1

  return vector_phrase

text_vectors = [text_to_vector(email, translator) for email in splitted_emails]

### Train, Test & Validate data
x = text_vectors
y = classifications['classificacao']

train_percentage = 0.8
train_size = train_percentage * len(y)
validation_size = len(y) - train_size

x_train = x[0:train_size]
y_train = y[0:train_size]

x_validation = x[train_size:]
y_validation = y[train_size:]


print(text_vectors)
