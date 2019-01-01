import numpy as np

class Utils(object):

    @classmethod
    def shape(cls, data):
        ''' Determine the shape of the data
        '''
        if isinstance(data, np.ndarray):
            return data.shape
        elif isinstance(data, list):
            rows = len(data)
            if rows == 0:
                raise Exception("Invalid 0-Dimensional Data")

            item = data[0]
            if isinstance(item, list):
                # Todo: consider recursion here?
                return (rows, len(item))
            else:
                # Assume the item is a scalar
                return (rows,)
        else:
            raise Exception("Unknown Data Format: %s" % type(data))
