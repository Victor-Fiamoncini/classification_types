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
mysterious_animal03 = [0, 0, 1]

# 1 = pig & -1 = dog
train_tags = [1, 1, 1, -1, -1, -1]
train_data = [pig01, pig02, pig03, dog01, dog02, dog03]
test_tags = [-1, 1, 1]
test_data = [mysterious_animal01, mysterious_animal02, mysterious_animal03]

model = MultinomialNB()
model.fit(train_data, train_tags)

prediction = model.predict(test_data)

differences = prediction - test_tags

hits = [d for d in differences if d == 0]
total_hits = len(hits)
total_test_elements = len(test_data)
hits_rate = (100 * total_hits) / total_test_elements

print(total_hits, total_test_elements, hits_rate)
