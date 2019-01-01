import math
import statistics
import scipy.stats

class Priori(object):
    ''' Helper methods to define a priori output distributions
    '''
    @classmethod
    def uniform(cls, values):
        ''' Assume that the output values follow a uniform priori distribution
        '''
        # Make sure we are only using unique values:
        vals = set(values)
        return lambda v: 1. / len(vals) if v in vals else 0.

    @classmethod
    def multinomial(cls, samples, frequencies=None, zero=0.0):
        ''' Assume that the output follows a multinomial distribution,
            using the sample given to determine distribution parameters

            Rmk: You can set a different value for zero, if you want to use an
                epsilon value to avoid automatically zeroing out the probability
        '''

        # Allow users to pass their own mapping of distribution frequencies
        frequencies = cls.frequencies(samples) if frequencies is None else frequencies

        def probability(outcomes):
            ''' Return the probbility that the outcomes were randomly selected
                from the multionomial distribution, defined above
            '''

            # Allow user to pass single items (for classificiation, for example):
            if not isinstance(outcomes, (list, set, tuple)):
                outcomes = [outcomes]

            counts = cls.counts(outcomes)
            prob = math.factorial(len(outcomes))
            for outcome, count in counts.iteritems():
                prob *= frequencies.get(outcome, zero) ** count / math.factorial(count)

            return prob

        return probability

    @classmethod
    def normal(cls, samples, mu=None, sigma=None):
        ''' Assume that the output follows a normal distribution,
            using the sample given to determine distribution parameters
        '''
        mu = statistics.mean(samples) if mu is None else mu
        sigma = statistics.stdev(samples) if sigma is None else sigma
        return lambda val: scipy.stats.norm.pdf(val, loc=mu, scale=sigma)

    @classmethod
    def counts(cls, items):
        ''' Get counts for each item
        '''
        counts = {}
        for item in items:
            counts.setdefault(item, 0)
            counts[item] += 1
        return counts

    @classmethod
    def frequencies(cls, items):
        ''' Get frequencies for each item
        '''
        counts = cls.counts(items)
        n = float(len(items))
        return {
            item: count / n
            for item, count in counts.iteritems()
        }


if __name__ == '__main__':
    uniform = Priori.uniform([1,2,3,4,5,6])
    print "Uniform A Priori:"
    print "%s: %.2f" % (3, uniform(3))
    print "%s: %.2f" % (0, uniform(0))

    multinomial = Priori.multinomial([1,2,3,4,5,6, 1, 2, 3, 2, 1, 2, 2,1, 1, 1, 1, 1, 1])
    print "Multinomial A Priori:"
    print "%s: %.2f" % ((3,), multinomial(3))
    print "%s: %.2f" % ((1, 1), multinomial([1, 1]))
    print "%s: %.2f" % ((0, 1), multinomial((0, 1)))
    print "%s: %.2f" % ((1, 2, 1), multinomial((1, 2, 1)))

    normal = Priori.normal([10, 11.2, 12, 11, 9, 10.4, 9, 12.1, 11, 8, 12, 9, 10, 10])
    print "Normal A Priori (From sample):"
    print "%s: %.2f" % (10.3, normal(10.3))
    print "%s: %.2f" % (6.6, normal(6.6))

    normal = Priori.normal(None, mu=0., sigma=1.)
    print "Normal A Priori (0,1):"
    print "%s: %.2f" % (0., normal(0))
    print "%s: %.2f" % (1., normal(1))
    print "%s: %.2f" % (2., normal(2))
