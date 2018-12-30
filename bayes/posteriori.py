class Posterior(object):
    ''' The Posterior distribution, represented by the P(B|A) in Bayes Theorem
    '''

    @classmethod
    def scalars(cls, inputs, outputs, zero=0.0):
        ''' Input Data is a scalar
        '''
        assert len(inputs) == len(outputs)
        counts = {}
        for i, output in enumerate(outputs):
            input = inputs[i]
            counts.setdefault(output, {}).setdefault(input, 0)
            counts[output][input] += 1

            counts[output].setdefault('_total', 0)
            counts[output]['_total'] += 1

        frequencies = {
            output: {
                item: float(count) / float(items['_total'])
                for item, count in items.iteritems()
            }
            for output, items in counts.iteritems()
        }

        return lambda input, output: frequencies.get(output, {}).get(input, zero)

    @classmethod
    def vectors(cls, inputs, outputs):
        ''' Input data is a list
        '''
        # todo
        pass

    @classmethod
    def dictionary(cls, inputs, outputs):
        ''' Input data is a dictionary
        '''
        # todo
        pass


if __name__ == '__main__':
    inputs = [1,2,3,4,5]
    outputs = [1,0,1,0,1]
    post = Posterior.scalars(inputs, outputs)
    print "Posterior Distribution of x%2 data on {1, 2, 3, 4, 5}"
    print "P(3|1) = %.2f" % post(3, 1)
    print "P(2|1) = %.2f" % post(2, 1)
    print "P(2|0) = %.2f" % post(2, 0)
