import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Minisom library and module is used for performing Self Organizing Maps
from minisom import MiniSom

""" ########## """
# SOM

# this csv file was created from code that exists in the final.py
# originally, i had the preprocessing code and csv in its own separate directory
# data = pd.read_csv('../preprocessing/punts_one_hot_encoded.csv')
# punts one hot encoded is located in final.py
data = punts_one_hot_encoded.drop(['gameId', 'playId', 'missedTackler', 'assistTackler', 'tackler', 'gunners', 'puntRushers',
'specialTeamsResult', 'season', 'gameDate', 'gameTimeEastern', 'homeTeamAbbr', 'visitorTeamAbbr',
'playDescription', 'possessionTeam', 'specialTeamsPlayType', 'specialTeamsResult', 'kickerId', 'returnerId', 'kickBlockerId',
'yardlineSide', 'penaltyJerseyNumbers', 'penaltyCodes', 'playResult',
'kickDirectionIntended_L', 'kickDirectionIntended_R', 'kickDirectionIntended_C',
'returnDirectionIntended_C', 'returnDirectionIntended_L', 'returnDirectionIntended_R',
'specialTeamsSafeties', 'vises', 'penaltyYards', 'passResult', 'return_team_min', 'gameClock', 'down'], axis=1)
# replace all nans in target variable with 0
data['kickReturnYardage'].fillna(0, inplace=True)
# target variable
y = data['kickReturnYardage'].values
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

n_bins = 5
bins = pd.cut(y, [-20, 0, 10, 20, 150], retbins=True)
encoded_y = label_encoder.fit_transform(bins[0])
# remove kick return yardage since it is target variable
x = data[data.columns.difference(['kickReturnYardage'])]
# data normalization
x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
x = x.values
# label_names = {1:'Negative Yards', 2:'Zero Yards', 3:'1-10', 4: '10-20', 5: '> 20'}
label_names = {0:'<= 0', 1:'0-10', 2:'10-20', 3: '> 20'}
colors = ['C0', 'C1', 'C2', 'C3', 'C4']
# Initialization and training
n_neurons = 12
m_neurons = 12
som = MiniSom(n_neurons, m_neurons, x.shape[1], sigma=1.5, learning_rate=0.5, 
              neighborhood_function='gaussian')

som.pca_weights_init(x)
som.train(x, 1000, random_order=True, verbose=True)  # random training

w_x, w_y = zip(*[som.winner(d) for d in x])
w_x = np.array(w_x)
w_y = np.array(w_y)

plt.figure(figsize=(10, 9))
plt.pcolor(som.distance_map().T, cmap='bone_r', alpha=.2)
plt.colorbar()

for c in np.unique(encoded_y):
    print(c)
    idx_target = y==c
    # print(idx_target)
    plt.scatter(w_x[idx_target]+.5+(np.random.rand(np.sum(idx_target))-.5)*.8,
                w_y[idx_target]+.5+(np.random.rand(np.sum(idx_target))-.5)*.8, 
                s=50, c=colors[c], label=label_names[c])
    # plt.scatter(w_x)
plt.legend(loc='upper right')
plt.grid()
# plt.savefig('resulting_images/som_seed.png')
plt.show()


# start off with cluster k (neurons) to 4?
# the MiniSom documentation said to do 12x12 since i have 701 samples?
"""
doing 12x12 with any number of iterations put quantization error above 3
then moving this up and quantization at 100,000 iterations put quantization down to 2

finally after moving nuerons to 50 with 100,000 iterations put quantization down below 1 (0.67)

50x50 with 10,000 iterations and sigma at 1.0 instead of 1.5 created more visual clusters
"""


# Initialization and training
n_neurons = 50
m_neurons = 50
som = MiniSom(n_neurons, m_neurons, x.shape[1], sigma=1.0, learning_rate=0.5, 
              neighborhood_function='gaussian', random_seed=0)

som.pca_weights_init(x)
som.train(x, 10000, random_order=True, verbose=True)  # random training

w_x, w_y = zip(*[som.winner(d) for d in x])
w_x = np.array(w_x)
w_y = np.array(w_y)

plt.figure(figsize=(10, 9))
plt.pcolor(som.distance_map().T, cmap='bone_r', alpha=.2)
plt.colorbar()

for c in np.unique(encoded_y):
    print(c)
    idx_target = y==c
    # print(idx_target)
    plt.scatter(w_x[idx_target]+.5+(np.random.rand(np.sum(idx_target))-.5)*.8,
                w_y[idx_target]+.5+(np.random.rand(np.sum(idx_target))-.5)*.8, 
                s=50, c=colors[c], label=label_names[c])
    # plt.scatter(w_x)
plt.legend(loc='upper right')
plt.grid()
# plt.savefig('resulting_images/som_seed.png')
plt.show()