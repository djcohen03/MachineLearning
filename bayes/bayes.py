''' Naive Bayes

    Bayes Theorem: P(A|B) = P(B|A) * P(A) / P(B)

'''
import time
from utils import Utils
from posterior import Posterior
from priori import Priori


class BayesClassifier(object):
    ''' Naive Bayes Classifier
            - inputs: 2-D lists or 2-D numpy.ndarrays (categorical)
            - outputs: 1-D list or 1-D numpy.ndarray (categorical)
    '''
    def __init__(self, priori, posterior, buckets):
        self.priori = priori
        self.posterior = posterior
        self.buckets = buckets

    @classmethod
    def uniform(cls, inputs, outputs, buckets=None, zero=0.0):
        ''' Create a Naive Bayes Classifier with a Uniform A Priori distribution
        '''
        # Determine the classifier's classes:
        buckets = set(outputs) if buckets is None else buckets

        print "Training Naive Bayes Classifier..."
        start = time.time()

        # Determine Priori and Posterior lambda functions:
        posterior = Posterior.immutable(inputs, outputs, zero=zero)
        priori = Priori.uniform(buckets)

        print "Finished Training Naive Bayes Classifier (%.2fs)" % (time.time() - start)

        # Create & Return classifier
        return cls(priori, posterior, buckets)

    @classmethod
    def multinomial(cls, inputs, outputs, buckets=None, frequencies=None, zero=0.0):
        ''' Create a Naive Bayes Classifier with a Multinomial A Priori distribution
        '''

        print "Training Naive Bayes Classifier..."
        start = time.time()

        # Determine Priori and Posterior lambda functions:
        posterior = Posterior.immutable(inputs, outputs, zero=zero)
        priori = Priori.multinomial(outputs, frequencies=frequencies, zero=zero)

        print "Finished Training Naive Bayes Classifier (%.2fs)" % (time.time() - start)

        # Determine the classifier's classes:
        buckets = set(outputs) if buckets is None else buckets

        # Create & Return classifier
        return cls(priori, posterior, buckets)

    def predict(self, input):
        ''' Using the given priori & posterior methods, return a classification
            method for classifying a given input as one of the buckets
        '''
        # Calculate the Bayes probabilities P(A|B) for each input feature value
        # (independently/naively)
        probs = {
            output: self.priori(output) * self.posterior(input, output)
            for output in self.buckets
        }
        total = sum(probs.values())

        # Normalize Result to account for P(B) constant denominator:
        if not total:
            # Just return uniform distribution here
            unif = 1. / len(self.buckets)
            return { output: unif for output in self.buckets }
        else:
            # Normalize Probabilities here
            return {
                output: probs.get(output) / total
                for output in self.buckets
            }

    def score(self, inputs, outputs):
        ''' Determine the performance on some testing inputs/outsputs
        '''
        inrows, features = Utils.shape(inputs)
        outrows, = Utils.shape(outputs)

        if inrows != outrows:
            raise Exception("Input/Output shape mismatch: %s in, %s out" % (inrows, outrows))

        success = []
        for i in range(inrows):
            predictions = [(output, p) for output, p in self.predict(inputs[i]).iteritems()]

            prediction, p = max(predictions, key=lambda item: item[1])
            correct = (prediction == outputs[i])
            success.append(int(correct))

        return float(sum(success)) / float(len(success))

if __name__ == '__main__':
    # Create Test Data:
    import random
    N = 5000
    inputs = [
        [random.randint(0, 40) for _ in range(10)]
        for _ in range(N)
    ]
    outputs = [input[1] % 2 for input in inputs]

    # Train Naive Bayes Classifier:
    nbc = BayesClassifier.uniform(inputs, outputs, zero=0.01)

    # Testing:
    print "\nTESTING"
    testinputs = [
        [random.randint(0, 40) for _ in range(10)]
        for _ in range(100)
    ]
    testoutputs = [input[1] % 2 for input in testinputs]
    score = nbc.score(testinputs, testoutputs)
    print "Testing Success Rate: %.2f%%" % (score * 100.)
