# TEXT MINING EN SOCIAL MEDIA
#### Master en Big Data Analytics - Curso 2015/2016
#### Universitat Politècnica de València

###### José Luis Cubero Somed and José Alberto Pérez Melián
---
#### Goal

Build a sentyment analysis tool to classify some tweets in six target classes `P+, P, NEU, N, N+, NONE` by following these steps:

1. Download the training tweets from [here](http://users.dsic.upv.es/~lhurtado/TMSM/concurso/TASS2014_training_polarity.txt)

2. Use some machine learning model to build the tweet sentyment classifier

3. Download the test tweets from [here](http://users.dsic.upv.es/~lhurtado/TMSM/concurso/TASS2014_test_ids.txt)

4. Predict the target class `POLARITY` by using the model developed in step 2

#### Result

Generate a text file with syntax `ID_TWEET \t POLARITY` with the prediction for all test tweets

---

#### Step 1 - Download the tweets

By using `Tweepy` we have downloaded both training and test tweets. Here there are some stats from both datasets:

- `TRAINING` dataset
   - File: `TASS2014_training_polarity.txt`
   - Total tweets: 7219
   - Existing tweets: 6950 (96%)
  
- `TEST` dataset
   - File: `TASS2014_test_ids.txt`
   - Total tweets: 60797
   - Existing tweets: 58044 (95%)
   
After all tweets has been downloaded, they are stored in `JSON` format in two different files (where each line represents a tweet)

#### Step 2 - Obtain text from tweets

Once we have all the tweets we proceeded to store their text in different files.

For `TRAINING` dataset, we store them in a file with header `ID_TWEET#####TEXT_TWEET#####POLARITY`

For `TEST` dataset, we store them in a file with header `ID_TWEET#####TEXT_TWEET`

#### Step 3
##### Step 3.1 - Tokenize training tweets

We use `TweetTokenizer` from `nltk.tokenize` to tokenize tweet's text:

```python
tokenizer = TweetTokenizer(strip_handles = True, reduce_len = True, preserve_case = False)
text_tok = tokenizer.tokenize(text_tweet)
```

- `strip_handles = True` removes username handles
- `reduce_len = True` normalizes word lengthening
- `preserve_case = False` downcases everything except emoticons

After that we remove hashtags:

```python
result = []
for i in range(len(text_tok)):
	if text_tok[i][0] != "#":
		result.append(text_tok[i])
```

Then we build a vocabulary from those tweets tokens and we generate a matrix of tf-idf tokens. 

```python
vec = TfidfVectorizer(norm = None, smooth_idf = False, max_features = 1000)
vocabulary = vec.fit(dataset_training_text)
X_train = vec.transform(dataset_training_text)
```

> In order to reduce the complexity of the problem we only take 1000 features.

##### Step 3.2 - Tokenize test tweets

We do the same as in step 3.1, but things change when we build the tf-idf tokens matrix.

We build the new tf-idf matrix by using the vocabulary used in step 3.1 for training tweets.

```python
X_test_labeled = vec.transform(dataset_test_labeled_text)
```

##### Step 3.3 - SVM model

We will use a Support Vector Machine model from `sklearn` with parameter `gamma = 0.001`

```python
classifier = svm.SVC(gamma = 0.001)
```

Then we train the model with training tweets

```python
classifier.fit(X_train[:], training_polarity[:])
```

Finally we predict the target variable `POLARITY` for test tweets

```python
predicted = classifier.predict(X_test_labeled[:])
```

#### Step 4

Once the prediction for test tweets has been made, we store it in a file `prediction_test.txt`

---

##### Notes

In order to test the model, we had a file `http://users.dsic.upv.es/~lhurtado/TMSM/concurso/TASS2014_minitest_polarity.txt` with 1000
test tweets labeled.

By using the `TRAINING` dataset and those test tweets labeled, we obtained an overall precision of `0.62` taking 1000 features.
