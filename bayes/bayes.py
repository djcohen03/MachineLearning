''' Naive Bayes

    Bayes Theorem: P(A|B) = P(B|A) * P(A) / P(B)

'''
from posterior import Posterior
from priori import Priori


class BayesClassifier(object):
    ''' Naive Bayes Classifier
    '''

    @classmethod
    def uniform(cls, inputs, outputs, buckets=None, zero=0.0):
        ''' Create a Naive Bayes Classifier with a Uniform A Priori distribution
        '''
        # Determine the classifier's classes:
        buckets = set(outputs) if buckets is None else buckets

        # Determine Priori and Posterior lambda functions:
        posterior = Posterior.immutable(inputs, outputs, zero=zero)
        priori = Priori.uniform(buckets)

        # Create & Return classifier
        return cls.classifier(priori, posterior, buckets)

    @classmethod
    def multinomial(cls, inputs, outputs, buckets=None, frequencies=None, zero=0.0):
        ''' Create a Naive Bayes Classifier with a Multinomial A Priori distribution
        '''
        # Determine Priori and Posterior lambda functions:
        posterior = Posterior.immutable(inputs, outputs, zero=zero)
        priori = Priori.multinomial(outputs, frequencies=frequencies, zero=zero)

        # Determine the classifier's classes:
        buckets = set(outputs) if buckets is None else buckets

        return cls.classifier(priori, posterior, buckets)

    @classmethod
    def classifier(cls, priori, posterior, buckets):
        ''' Using the given priori & posterior methods, return a classification
            method for classifying a given input as one of the buckets
        '''
        def classify(input):
            probs = {
                bucket: priori(bucket) * posterior(input, bucket)
                for bucket in buckets
            }
            # Normalize Probabilities (so they sum to one):
            total = sum(probs.values())
            if not total:
                # All probabilities sum to zero, so return uniform distribution
                return {
                    bucket: 1. / len(buckets)
                    for bucket, prob in probs.iteritems()
                }
            else:
                return {
                    bucket: prob / total
                    for bucket, prob in probs.iteritems()
                }

        return classify


if __name__ == '__main__':
    inputs = [(i, i**2) for i in range(30)]
    outputs = [i % 2 for i, j in inputs]
    classify = BayesClassifier.uniform(inputs, outputs)
    print "(5, 25): %s" % classify((5, 25))
    print "(5, 26): %s" % classify((5, 26))
