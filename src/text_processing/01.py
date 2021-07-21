from os import getcwd
import pandas as pd
from past.builtins import xrange

### Get data & separate train/test data
file_path = getcwd() + '/src/text_processing/data/emails.csv'
classifications = pd.read_csv(file_path)

raw_emails = classifications['email']
splitted_emails = raw_emails.str.lower().str.split(' ')

all_words = set()
for word_list in splitted_emails:
  all_words.update(word_list)

all_words_total = len(all_words)
all_words_tuples = list(zip(all_words, range(all_words_total)))


email01 = "Se eu comprar cinco anos antecipados, eu ganho algum desconto?"
email02 = "O exercício 15 do curso de Java 1 está com a resposta errada. Pode conferir pf?"
email03 = "Existe algum curso para cuidar do marketing da minha empresa?"

print(all_words_tuples, all_words_total)
