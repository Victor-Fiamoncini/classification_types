from sklearn.naive_bayes import MultinomialNB

# Is fat; Has short legs; He barks;
pig01 = [1, 1, 0]
pig02 = [1, 1, 0]
pig03 = [1, 1, 0]

dog01 = [1, 1, 1]
dog02 = [0, 1, 1]
dog03 = [0, 1, 1]

mysterious_animal01 = [1, 1, 1]
mysterious_animal02 = [1, 0, 0]

# 1 = pig & -1 = dog
tags = [1, 1, 1, -1, -1, -1]
train_data = [pig01, pig02, pig03, dog01, dog02, dog03]

# Data to predict (test)
test = [mysterious_animal01, mysterious_animal02]

model = MultinomialNB()
model.fit(train_data, tags)

prediction = model.predict(test)

print(prediction)
