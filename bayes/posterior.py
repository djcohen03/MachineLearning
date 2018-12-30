class Posterior(object):
    ''' The Posterior distribution, represented by the P(B|A) in Bayes Theorem
    '''

    @classmethod
    def immutable(cls, inputs, outputs, zero=0.0):
        ''' Input Data is a immutable (and therefore _is_ hashable)
        '''
        if len(inputs) != len(outputs):
            raise Exception("Input/Output Lengths Mismatch (%s, %s)" % (len(input), len(outputs)))

        # Gather counts of inputs, given a particular output
        counts = {}
        for i, output in enumerate(outputs):
            input = inputs[i]
            counts.setdefault(output, {}).setdefault(input, 0)
            counts[output][input] += 1

            counts[output].setdefault('_total', 0)
            counts[output]['_total'] += 1

        # Determine each input's frequency, based on the given output
        frequencies = {
            output: {
                item: float(count) / float(items['_total'])
                for item, count in items.iteritems()
            }
            for output, items in counts.iteritems()
        }

        return lambda input, output: frequencies.get(output, {}).get(input, zero)

    @classmethod
    def dictionary(cls, inputs, outputs, zero=0.0):
        ''' Input data is a dictionary (todo)
        '''
        pass

    @classmethod
    def lists(cls, inputs, outputs, zero=0.0):
        ''' Input data points are lists (todo)
        '''
        pass


if __name__ == '__main__':
    inputs = [1,2,3,4,5]
    outputs = [1,0,1,0,1]
    post = Posterior.immutable(inputs, outputs)
    print "Posterior Distribution of x%2 data on {1, 2, 3, 4, 5}"
    print "P(3|1) = %.2f" % post(3, 1)
    print "P(2|1) = %.2f" % post(2, 1)
    print "P(2|0) = %.2f" % post(2, 0)

    inputs = [(1, 1), (2, 1), (3, 1), (2, 2), (4, 1)]
    outputs = [0, 1, 0 , 0, 1]
    post = Posterior.immutable(inputs, outputs)
    print "Posterior Distribution of (x + y)%%2 data on %s" % inputs
    print "P((3, 1)|0) = %.2f" % post((3, 1), 0)
    print "P((3, 1)|1) = %.2f" % post((3, 1), 1)
    print "P((2, 1)|1) = %.2f" % post((2, 1), 1)
    print "P((2, 0)|0) = %.2f" % post((2, 0), 0)
    print "P(2|0) = %.2f" % post(2, 0)
