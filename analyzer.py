import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier

TRAINFILE = 'train.csv'
TESTFILE = 'test.csv'
OUTFILE = 'predictions.csv'
NESTIMATORS = 100

def read_file(filename):
    with open(filename, 'rb') as buff:
        buff.next()
        the_data = np.array([map(int, row.split(",")) for row in buff])
    return the_data

def train_and_predict():
    train_data, test_data = map(read_file, [TRAINFILE,TESTFILE])

    forest = RandomForestClassifier(n_estimators = NESTIMATORS).fit(train_data[:, 1:], train_data[:, 0])

    output = Forest.predict(test_data)

    with open(OUTFILE, 'wb') as buff:
        buff.write("ImageId,Label\n")
        buff.write("\n".join(["{:d},{:d}".format(ndx+1,row) for ndx, row in enumerate(output)]))

if __name__ == '__main__':
    sys.exit(train_and_predict())
