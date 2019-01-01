class DataSets(object):
    ''' Here we have a handful of data sets
    '''

    @classmethod
    def boston(cls):
        ''' Features:
                - CRIM     per capita crime rate by town
                - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
                - INDUS    proportion of non-retail business acres per town
                - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
                - NOX      nitric oxides concentration (parts per 10 million)
                - RM       average number of rooms per dwelling
                - AGE      proportion of owner-occupied units built prior to 1940
                - DIS      weighted distances to five Boston employment centres
                - RAD      index of accessibility to radial highways
                - TAX      full-value property-tax rate per $10,000
                - PTRATIO  pupil-teacher ratio by town
                - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
                - LSTAT    % lower status of the population
                - MEDV     Median value of owner-occupied homes in $1000's
            Output: House Price
        '''
        from sklearn.datasets import load_boston
        return load_boston()

    @classmethod
    def diabetes(cls):
        ''' Features:
                - Age
                - Sex
                - Body mass index
                - Average blood pressure
                - S1
                - S2
                - S3
                - S4
                - S5
                - S6
            Output: quantitative measure of disease progression one year after baseline
        '''
        from sklearn.datasets import load_diabetes
        return load_diabetes()

    @classmethod
    def cancer(cls):
        ''' Features:
                - radius (mean of distances from center to points on the perimeter)
                - texture (standard deviation of gray-scale values)
                - perimeter
                - area
                - smoothness (local variation in radius lengths)
                - compactness (perimeter^2 / area - 1.0)
                - concavity (severity of concave portions of the contour)
                - concave points (number of concave portions of the contour)
                - symmetry
                - fractal dimension ("coastline approximation" - 1)

                The mean, standard error, and "worst" or largest (mean of the three
                largest values) of these features were computed for each image,
                resulting in 30 features.  For instance, field 3 is Mean Radius, field
                13 is Radius SE, field 23 is Worst Radius.
            Output: [WDBC-Malignant, WDBC-Benign]
        '''
        from sklearn.datasets import load_breast_cancer
        return load_breast_cancer()

    @classmethod
    def digits(cls):
        ''' Input: 8x8 images of digits (where each pixel is a number {0,...,16})
            Output: Digit
        '''
        from sklearn.datasets import load_digits
        return load_digits()

    @classmethod
    def iris(cls):
        ''' Features:
                - sepal length in cm
                - sepal width in cm
                - petal length in cm
                - petal width in cm
            Output: [Iris-Setosa, Iris-Versicolour, Iris-Virginica]
        '''
        from sklearn.datasets import load_iris
        return load_iris()

    @classmethod
    def wine(cls):
        ''' Features:
                - Alcohol
                - Malic acid
                - Ash
                - Alcalinity of ash
                - Magnesium
                - Total phenols
                - Flavanoids
                - Nonflavanoid phenols
                - Proanthocyanins
                - Color intensity
                - Hue
                - OD280/OD315 of diluted wines
                - Proline
            Output: [class_0, class_1, class_2]
        '''
        from sklearn.datasets import load_wine
        return load_wine()

    @classmethod
    def linnerud(cls):
        ''' Features: ['Chins', 'Situps', 'Jumps']
            Output: ['Weight', 'Waist', 'Pulse']
        '''
        from sklearn.datasets import load_linnerud
        return load_linnerud()
