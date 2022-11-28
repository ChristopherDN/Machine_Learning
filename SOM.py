import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show

dataset = pd.read_csv('Credit_Card_Applications.csv')  # in this dataset, class=0 means application was rejected. Class = 1 means approved.
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

# x and y are dimensions of the Map. input_len is number of columns in X.
# sigma is the radius from the data point. Nodes inside that circle will be adjusted towards the winning node.

som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

bone()  # create empty window
pcolor(som.distance_map())  # plots mean inter-neuron distance from each node to its neighbors !
colorbar()  # legend for each axis
markers = ['o', 's']  # 'o' = circle, 's' = square
colors = ['r', 'g']  # 'r' = red, 'g' = green
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],  # if y=0, the application was rejected, and it will print a circle. Etc.
         markeredgecolor=colors[y[i]],  # if y=0, it will print a red circle. Otherwise green square.
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)
show()

mappings = som.win_map(X)
frauds = np.concatenate((mappings[(4, 3)], mappings[(2, 3)]),
                        axis=0)  # these values are just examples. You have to find the one or two
# most bright/white squares on your plot, since each map is different.
frauds = sc.inverse_transform(frauds)

print('Fraud Customer IDs')
for i in frauds[:, 0]:
    print(int(i))
