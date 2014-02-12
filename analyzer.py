import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier

csv_train_file = csv.reader(open('train.csv', 'rb'))
csv_test_file = csv.reader(open('test.csv', 'rb'))

csv_train_file.next()
csv_test_file.next()

train_data = []
for row in csv_train_file:
    train_data.append(row)

test_data = []
for row in csv_test_file:
    test_data.append(row)

train_data = np.array(train_data)
test_data = np.array(test_data)

Forest = RandomForestClassifier(n_estimators = 100)

Forest = Forest.fit(train_data[0:, 1:], train_data[0:, 0])

Output = Forest.predict(test_data[0:, :])

outfile = csv.writer(open('predictions.csv', 'wb'))

outfile.writerow(["ImageId", "Label"])

for ndx, row in enumerate(Output):
    outfile.writerow([ndx + 1, row])
