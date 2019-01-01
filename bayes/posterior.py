import numpy as np
from utils import Utils

class Posterior(object):
    ''' The Posterior distribution, represented by the P(B|A) in Bayes Theorem
    '''

    @classmethod
    def immutable(cls, inputs, outputs, zero=0.0):
        ''' Input Data is a immutable (and therefore _is_ hashable)
        '''

        inrows, features = Utils.shape(inputs)
        outrows, = Utils.shape(outputs)

        if inrows != outrows:
            raise Exception("Input/Output Lengths Mismatch (%s in, %s out)" % (inrows, outrows))

        # Create a feature-wise input frequency tracker
        counts = [{} for _ in range(features)]

        # Gather counts of inputs, given a particular output
        for i, output in enumerate(outputs):
            input = inputs[i]
            for j, featureval in enumerate(input):
                # Add this input/output feature pair to the counts data container
                counts[j].setdefault(output, {})
                counts[j][output].setdefault(featureval, 0)
                counts[j][output][featureval] += 1
                counts[j][output].setdefault('_total', 0)
                counts[j][output]['_total'] += 1

        frequencies = [{} for _ in range(features)]
        for i, featuredata in enumerate(counts):
            for output, inputcounts in featuredata.iteritems():
                frequencies[i].setdefault(output, {})
                _total = float(inputcounts['_total'])
                for input, inputcount in inputcounts.iteritems():
                    frequencies[i][output][input] = inputcount / _total

        def probability(inputs, output):
            ''' Define the P(B|A) frequentist function for Bayes posterior evaluation
            '''
            probs = [frequencies[i].get(output, {}).get(inputs[i], zero) for i in range(features)]
            return np.prod(probs)

        return probability

if __name__ == '__main__':
    # Example problem
    import random
    N = 5000
    inputs = [
        [random.randint(0, 40) for _ in range(4)]
        for _ in range(N)
    ]
    outputs = [input[1] % 2 for input in inputs]
    post = Posterior.immutable(inputs, outputs, zero=0.01)
    print "Posterior Distribution Example, for output = input[1] % 2"
    input = [1, 3, 3, 2]
    print "Input: %s" % input
    print "P(%s|0) = %.9f" % (input, post(input, 0))
    print "P(%s|1) = %.9f" % (input, post(input, 1))
