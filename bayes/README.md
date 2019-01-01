# Naive Bayes
This repository holds a Naive Bayes classification training/predicting class, inspired by the excellent `scikit-learn` [package](https://scikit-learn.org/stable/)! Please read: [Naive Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)

## How-To:
A good workflow might look something like this:
1. Obtain some 2-D input and 1-D output data
2. Choose an a priori output distribution:
    - Choose **Uniform** if the outputs are expected to be roughly uniform
    - Choose **Multinomial** if we suspect that the output distribution is _nonuniform_ (see: [Multinomial distribution](https://en.wikipedia.org/wiki/Multinomial_distribution))
3. Create/Train the classifier with the class methods `BayesClassifier.uniform` or `BayesClassifier.multinomial`

And that should be it! The result should be an object capable of predicting outputs with the `nbc.predict(input)` prediction method!

## Class Documentation

### BayesClassifier:
```
class BayesClassifier():
    __init__(priori, posterior, buckets)
    ''' Initialize NBC class with the given prior and posterior distributions, and with the given outputs'''

    predict(input)
    ''' Use the trained model to predict the output for the given input '''

    score(inputs, outputs)
    ''' Test a set of inputs, to see if they match the given outputs, return % success '''

    @classmethod
    uniform(inputs, outputs, buckets=None, zero=0.0)
    ''' Initialize NBC with a uniform a priori assumtion'''

    @classmethod
    multinomial(inputs, outputs, buckets=None, frequencies=None, zero=0.0)
    ''' Initialize NBC with a multinomial a priori assumtion'''

```

### Priori:
```
class Priori():
    @classmethod
    def uniform(values):
    ''' Given a 1-D list of outputs, return an a priori lamdba function based on the uniform distribution '''

    @classmethod
    def multinomial(samples, frequencies=None, zero=0.0):
    ''' Given a 1-D list of outputs, return an a priori lamdba function based on the multinomial distribution '''

```

### Posterior:
```
class Posterior():
    @classmethod
    def immutable(inputs, outputs, zero=0.0):
    ''' Given a 2-D list of immutable inputs, and a 1-D list of outputs, train the frequentist posterior distribution lambda function '''
```



## Example:
Here we whip up a simple example where the `output` is the value of `input[1] % 2`. (Note that we limit the inputs to integers between 0 and 40).  As you will see, the classifier does a nice job of **implicitely learning this simple rule**, with as little as 1000 data points. See:

```
import random
from bayes import BayesClassifier

# Create Test Data:
N = 1000
inputs = [[random.randint(0, 40) for _ in range(10)] for _ in range(N)]
outputs = [input[1] % 2 for input in inputs]

# Create & Train Naive Bayes Classifier:
nbc = BayesClassifier.uniform(inputs, outputs, zero=0.01)

# Test Our Classifier:
print "\nTESTING:"
testinputs = [[random.randint(0, 40) for _ in range(10)] for _ in range(100)]
testoutputs = [input[1] % 2 for input in testinputs]
score = nbc.score(testinputs, testoutputs)
print "Testing Success Rate: %.2f%%" % (score * 100.)
```

## What's next?
Some ideas for what's next here:
- Let's think about adding in functionality for making continuous/regression predictions
- Might be nice to have an explicit _Bernoulli_ distribution wrapper, even though it's just a special case of the Multinomial distribution
- Let's also think about adding other potential classification-based a priori distribution options
- Write some unit tests
- Not sure if our project will work on input data with greater than 2 dimensions, or if that would even make sense in this context.  A next step might be to explore higher dimensional input data
