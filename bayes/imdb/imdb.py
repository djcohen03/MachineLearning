import numpy as np
from keras.datasets import imdb as imdb_


# Dictionary Cutoff
MAX_WORDS = 10000


class Methods(object):

    wordindex = None
    wordmapping = None
    @classmethod
    def loadwords(cls):
        start = time.time()
        print("Loading IMDB Word Bank")
        cls.wordindex = imdb_.get_word_index()
        print("Loaded IMDB Word Bank in %.2fs" % (time.time() - start))
        cls.wordmapping = dict([(v, k) for (k,v) in cls.wordindex.items()])

    @classmethod
    def decode(cls, words):
        if not cls.wordindex or not cls.wordmapping:
            cls.loadwords()
        # Decode a word index to the string representation:
        return ' '.join([cls.wordmapping.get(i - 3, '?') for i in words])

    @classmethod
    def encode(cls, words):
        if not cls.wordindex or not cls.wordmapping:
            cls.loadwords()
        wordlist = words.lower().split(' ')
        indexlist = [cls.wordindex.get(word) for word in wordlist]
        return [x for x in indexlist if x is not None]

    @classmethod
    def ergprediction(cls, val):
        return "Positive" if val > 0.8 else "Negative" if val < 0.2 else "Unsure"

    @classmethod
    def vectorize(cls, samples, dimensions=MAX_WORDS):
        ''' Words are represented as indicies from 1 to MAX_WORDS, so this will
            transform an array of word indicies (eg [543, 33, 1435, 341, 2,...]) to a
            binary boolean array representing the the presense of the word in the
            array of indicies (eg [0, 1, 0, 0,...])
            This is done to the input data because it standardizes the shape/size of the
            input arrays to be (MAX_WORDS,)

            My Note: This might be improved by _not_ destroying the information
            corresponding to duplicate words in one array.  One might solve this by
            modifying the input above to use "counts". This would mean instead of
            using a simple "1" for presence and "0" for absense, we would use "n"
            for the number of times the word occurs in the sample text- this way we
            would know how many times each word appears.  Doesn't seem important
            enough to have been metioned in the book, however, so might just be a
            negligable improvement
        '''
        # Initialize an all-zeros matrix to represent the entire sample set
        vectorized = np.zeros((len(samples), dimensions))
        for i, sample in enumerate(samples):
            # Vectorize the ith sample:
            vectorized[i, sample] = 1.
        return vectorized




if __name__ == '__main__':

    # Load IMDB Training Data (With 10,000 most common words)
    print("Loading IMDB Datasset")
    start = time.time()
    (traindata, trainlabels), (testdata, testlabels) = imdb_.load_data(num_words=MAX_WORDS)
    print("Loaded IMDB Dataset (in %.2fs)" % (time.time() - start))
    # Vectorize the input data (see explanation above),
    vectortized_traindata = Methods.vectorize(traindata)
    trainlabels = np.array(trainlabels)

    # Todo: train Naive Bayes classifier on training data set



    # print("IMDB Samples:")
    # count = 10
    # testindicies = random.sample(range(len(testdata)), k=count)
    # tests = testdata[testindicies]
    # vectorized_tests = Methods.vectorize(tests)
    # predictions = network.predict(vectorized_tests)
    # for i in range(count):
    #     prediction = Methods.ergprediction(predictions[i])
    #     expected = Methods.ergprediction(testlabels[testindicies[i]])
    #     words = Methods.decode(tests[i])
    #     print("Predicted: %s\tActual: %s\tSample: %s" % (
    #         prediction,
    #         expected,
    #         words[:100]
    #     ))
