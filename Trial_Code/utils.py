import numpy as np

def derivative(data, dt):
    data_shift = np.roll(data, 1, axis=0)
    data_shift[0] = data[0]
    #print('data_shift :', data_shift)
    return (data - data_shift)/dt

class DynamicRecArray(object):
    def __init__(self, dtype):
        self.dtype = np.dtype(dtype)
        self.length = 0
        self.size = 10
        self._data = np.empty(self.size, dtype=self.dtype)

    def __len__(self):
        return self.length

    def append(self, rec):
        if self.length == self.size:
            self.size = int(1.5*self.size)
            self._data = np.resize(self._data, self.size)
        self._data[self.length] = rec
        self.length += 1

    def extend(self, recs):
        for rec in recs:
            self.append(rec)

    @property
    def data(self):
        return self._data[:self.length]