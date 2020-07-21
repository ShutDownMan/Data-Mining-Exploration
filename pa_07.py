import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X = []
Y = []

# import the data
data = pd.read_csv('./data/boystown.csv', sep=' ');

#print(data);

data['sex'] -= 1;
data['dadjob'] -= 2;
data['momjob'] -= 2;

def map_gpa(gpa):
	if(gpa >= 3):
		return 0;
	return 1;

data['gpa'] = [map_gpa(gpa) for gpa in data['gpa']];

#print(data['gpa']);

def normalize(x):
	x = np.array(x);
	return ((x - min(x)) / (max(x)-min(x))).tolist();

for label, content in data.items():
	data[label] = normalize(content);

#print(data);

# shuffle
data = data.sample(frac=1).reset_index(drop=True);


X = data.drop('gpa', axis=1);
Y = data['gpa'].values;

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25);

print('K\tTrain\tTest');
evalArray = [];
for k in range(2, 15):
	model = KNeighborsClassifier(n_neighbors=k);
	model.fit(X, Y);

	trainScore = model.score(Xtrain, Ytrain);
	testScore = model.score(Xtest, Ytest);
	evalArray.append((k, trainScore, testScore));

	#print('Train score =', model.score(Xtrain, Ytrain));
	#print('Test score =', model.score(Xtest, Ytest));
	#print('----------');


evalArray = np.array(evalArray, dtype=[('k', int), ('train', float), ('test', float)]);

evalArray.sort(axis=0, order='test');
#print(evalArray);

#print(evalArray.tolist());
evalArray = evalArray[::-1];

for k, trainScore, testScore in evalArray.tolist():
	print(k,
		'\t', "{:.2}".format(trainScore),
		'\t', "{:.2}".format(testScore));
