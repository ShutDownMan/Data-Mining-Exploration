import pandas as pd
import numpy as np

X = []
Y = []

# import the data
data = pd.read_csv('./data/boystown.csv', sep=' ');

#print(data);

data['sex'] -= 1;

print(data);