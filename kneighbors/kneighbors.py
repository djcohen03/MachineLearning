import scipy.stats
import numpy as np


class KNeighbors(object):
    ''' K-Nearest Neighbors Classifier
            - inputs: 2-D lists or 2-D numpy.ndarrays (continuous)
            - outputs: 1-D list or 1-D numpy.ndarray (categorical)
    '''
    def __init__(self, inputs, outputs, distance, k=1):
        self.inputs = inputs
        self.outputs = outputs
        self.distance = distance
        self.k = k

    def predict(self, input):
        ''' Predict the class of the input as the most common of its k
            nearest neighbors
        '''
        distances = [
            (self.distance(input, point), index)
            for index, point in enumerate(self.inputs)
        ]
        nearest = sorted(distances, key=lambda item: item[0])[:self.k]

        classes = [self.outputs[index] for _, index in nearest]
        return scipy.stats.mode(classes).mode[0]

    def score(self, inputs, outputs):
        ''' Determine the performance on some testing inputs/outsputs
        '''
        inputs = np.array(inputs)
        outputs = np.array(outputs)
        inrows, features = inputs.shape
        outrows, = outputs.shape

        if inrows != outrows:
            raise Exception("Input/Output shape mismatch: %s in, %s out" % (inrows, outrows))

        success = []
        for i in range(inrows):
            prediction = self.predict(inputs[i])
            correct = (prediction == outputs[i])
            success.append(int(correct))

        return float(sum(success)) / float(len(success))

    @classmethod
    def make(cls, inputs, outputs, k=1):
        ''' Inplicitely determine distance function
        '''
        # Convert both inputs and outputs to numpy arrays (if not already):
        inputs = np.array(inputs)
        outputs = np.array(outputs)

        def distance(input1, input2):
            return abs(np.linalg.norm(input1 - input2))

        return cls(inputs, outputs, distance, k=k)




if __name__ == '__main__':
    # Make some training data:
    import random
    import time
    N = 100
    inputs = [[random.randint(0, 40) for _ in range(5)] for _ in range(N / 2)]
    inputs += [[random.randint(20, 60) for _ in range(5)] for _ in range(N / 2)]
    outputs = [0] * (N / 2) + [1] * (N / 2)

    # Create KNN classifier:
    print "Creating K Nearest Neighbors Classifier"
    knn = KNeighbors.make(inputs, outputs, k=10)

    # Create some testing data:
    testinputs = [[random.randint(0, 40) for _ in range(5)] for _ in range(50)]
    testinputs += [[random.randint(20, 60) for _ in range(5)] for _ in range(50)]
    testoutputs = [0] * 50 + [1] * 50

    print "TESTING"
    start = time.time()
    score = knn.score(testinputs, testoutputs)
    print "Testing Success Rate: %.2f%% (%.2fs)" % (score * 100., time.time() - start)
