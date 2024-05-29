import matplotlib.pyplot as plt
from kalgo import KMeans
import pandas as pd

df=pd.read_csv('playerperformance/player_rankings_2024.csv')
X = df[['RAA', 'Wins']].to_numpy()

km = KMeans(n_clusters=3,max_iter=100)

y_means=km.fit_predict(X)

plt.scatter(X[y_means == 0,0],X[y_means == 0,1],color='red')
plt.scatter(X[y_means == 1,0],X[y_means == 1,1],color='blue')
plt.scatter(X[y_means == 2,0],X[y_means == 2,1],color='green')

plt.show() 