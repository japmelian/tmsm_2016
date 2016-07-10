# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import TweetTokenizer
from sklearn import datasets, svm, metrics
import numpy
import datetime

######### TRAINING PART #########
inputFile = open("training_tweets_text_polarity.txt", "r")

dataset_training_id = []
dataset_training_text = []
dataset_training_polarity = []

for line in inputFile:
	id_tweet, text_tweet, polarity_tweet = line.strip().split("#####")

	dataset_training_id.append(id_tweet)
	dataset_training_polarity.append(polarity_tweet)

	tokenizer = TweetTokenizer(strip_handles = True, reduce_len = True, preserve_case = False)

	text_tok = tokenizer.tokenize(text_tweet)

	result = []

	for i in range(len(text_tok)):
		if text_tok[i][0] != "#":
			result.append(text_tok[i])

	dataset_training_text.append(" ".join(result))

training_polarity = numpy.array(dataset_training_polarity)

vec = TfidfVectorizer(norm = None, smooth_idf = False, max_features = 1000)
vocabulary = vec.fit(dataset_training_text)
X_train = vec.transform(dataset_training_text)

del dataset_training_id
del dataset_training_text
del dataset_training_polarity

######### TEST (LABELED) PART #########
inputFile = open("test_tweets_text.txt", "r")

dataset_test_labeled_id = []
dataset_test_labeled_text = []
dataset_test_labeled_polarity = []

for line in inputFile:
	id_tweet, text_tweet = line.strip().split("#####")[0:2]

	dataset_test_labeled_id.append(id_tweet)
	#dataset_test_labeled_polarity.append(polarity_tweet)

	tokenizer = TweetTokenizer(strip_handles = True, reduce_len = True, preserve_case = False)

	text_tok = tokenizer.tokenize(text_tweet)

	result = []

	for i in range(len(text_tok)):
		if text_tok[i][0] != "#":
			result.append(text_tok[i])

	dataset_test_labeled_text.append(" ".join(result))

X_test_labeled = vec.transform(dataset_test_labeled_text)

del dataset_test_labeled_text


########################### MODEL ###########################
X_train = X_train.toarray()
X_test_labeled = X_test_labeled.toarray()


print "  Creating model..."
start_time = datetime.datetime.now()

classifier = svm.SVC(gamma = 0.001)

classifier.fit(X_train[:], training_polarity[:])
print "  Model created!"
end_time = datetime.datetime.now()

print end_time - start_time

print "  Making prediction..."
predicted = classifier.predict(X_test_labeled[:])
print "  Prediction done!"
end_time = datetime.datetime.now()

print end_time - start_time

print type(predicted)

#print("Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(expected, predicted)))

outputFile = open("prediction_text.txt", "w")

for i in range(len(predicted)):
	outputFile.write(dataset_test_labeled_id[i] + "\t" + predicted[i] + "\n")

outputFile.close()