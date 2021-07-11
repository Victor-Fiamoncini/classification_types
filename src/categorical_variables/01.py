from os import getcwd
import pandas as pd

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

print(train_size, test_size)
