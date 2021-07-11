from os import getcwd
import pandas as pd

file_path = getcwd() + '/src/multi_category/data/users.csv'

data_frame = pd.read_csv(file_path)

x = data_frame[['home', 'busca', 'logado']]
y = data_frame['comprou']

x_dummies = pd.get_dummies(x)
y_dummies = y
