# packages
import pandas as pd
import numpy as np

# from football_functions import create_football_field
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# data
plays = pd.read_csv('../plays.csv')
games = pd.read_csv('../games.csv')
tracking = pd.read_csv('../tracking2020.csv')
players = pd.read_csv('../players.csv')
pff = pd.read_csv('../pff.csv')

# functions
import math
def euclidean_distance(a, b):
    x1, y1 = a
    x2, y2 = b
    x = x2 - x1
    y = y2 - y1
    return math.sqrt(x**2 + y**2)

"""####################"""
#############
# PFF EDA #
mask = (pff['kickType'] == 'N') | (pff['kickType'] == 'R') | (pff['kickType'] == 'A')
# kickoffs
kickoffs = pff[~mask]

# punts
punts = pff[mask]

fig, axs = plt.subplots(2,2)
fig.suptitle('Kickoffs')
axs[0, 0].bar(['C', 'L', 'R'], kickoffs['kickDirectionIntended'].value_counts())
axs[0, 0].set_title('Intended Kick Direction')

axs[0, 1].bar(['C', 'L', 'R'], kickoffs['kickDirectionActual'].value_counts())
axs[0, 1].set_title('Actual Kick Direction')

axs[1, 0].bar(['C', 'L', 'R'], kickoffs['returnDirectionIntended'].value_counts())
axs[1, 0].set_title('Intended Return Direction')

axs[1, 1].bar(['C', 'L', 'R'], kickoffs['returnDirectionActual'].value_counts())
axs[1, 1].set_title('Actual Return Direction')

fig.tight_layout()

fig, axs = plt.subplots(2,2)
fig.suptitle('Punts')
axs[0, 0].bar(['C', 'L', 'R'], punts['kickDirectionIntended'].value_counts())
axs[0, 0].set_title('Intended Kick Direction')

axs[0, 1].bar(['C', 'L', 'R'], punts['kickDirectionActual'].value_counts())
axs[0, 1].set_title('Actual Kick Direction')

axs[1, 0].bar(['C', 'L', 'R'], punts['returnDirectionIntended'].value_counts())
axs[1, 0].set_title('Intended Return Direction')

axs[1, 1].bar(['C', 'L', 'R'], punts['returnDirectionActual'].value_counts())
axs[1, 1].set_title('Actual Return Direction')

fig.tight_layout()

"""####################"""
#############
# plays EDA #
kick_off_plays = plays[plays['specialTeamsPlayType'] == 'Kickoff']
punt_plays = plays[plays['specialTeamsPlayType'] == 'Punt']

kickoff_kry = kick_off_plays['kickReturnYardage']
punt_kry = punt_plays['kickReturnYardage']

fig, axs = plt.subplots(1, 2)
fig.tight_layout()
axs[0].hist(kickoff_kry, bins=30)
axs[1].hist(punt_kry, bins=30)
axs[0].set_title("Kickoff Kick Return Yardage")
axs[0].set_xlabel("Yards")
axs[1].set_title("Punt Kick Return Yardage")
axs[1].set_xlabel("Yards")

kickoff_kl = kick_off_plays['kickLength']
punt_kl = punt_plays['kickLength']

fig, axs = plt.subplots(1, 2)
axs[0].hist(kickoff_kl, bins=30)
axs[1].hist(punt_kl, bins=30)
axs[0].set_title("Kickoff Kick Kick Length")
axs[0].set_xlabel("Yards")
axs[1].set_title("Punt Kick Length")
axs[1].set_xlabel("Yards")

box_whisker_df = pd.Series([kickoff_kry, punt_kry], index=['Kickoff Kick Return Yardage', 'Punt Kick Return Yardage'])
box_whisker_df = pd.concat([kickoff_kry, punt_kry], axis=1, names=['Kickoff', 'Punt'])
box_whisker_df.columns = ['Kickoff', 'Punt']
box_whisker_df.plot.box(title="Kick Return Yardage", grid=True)

"""####################"""

################################################################
# created new dataframe and added tracking engineered features #
punt_plays_features = pd.DataFrame()
games_2020 = games[games['season'] == 2020]
# select all plays data that match 2020 the game ids
plays_2020 = plays[plays['gameId'].isin(games_2020['gameId'].unique())]
# merging plays and game data on game id as key
game_and_plays = pd.merge(games, plays, on="gameId")
# remove unnecessary columns or columns i think we dont need
game_and_plays = game_and_plays.drop(['gameTimeEastern', 'passResult', 'absoluteYardlineNumber', 'quarter', 'down', 'yardsToGo'], axis=1)
# filter out plays dataset
# find all plays that were kickoffs and punts - get it from specialTeamsPlayType
plays_2020 = game_and_plays[game_and_plays['season'] == 2020]
punts_2020 = plays_2020.query('specialTeamsPlayType == "Punt" and specialTeamsResult == "Return"')
tracking_2020_punt_received = tracking[tracking['event'] == 'punt_received']
# joining on multiple keys
punts_2020 = pd.merge(punts_2020, tracking_2020_punt_received, on=['gameId', 'playId'])
# remove football tracking data
punts_2020 = punts_2020[punts_2020['team'] != 'football']
# dataframe columns
data_columns = punts_2020.columns.to_list()
# instantiating a new dataframe that i will append these groups to
new_punts_dataframe = pd.DataFrame(columns=data_columns)
# remove plays where there is no kick returner and plays with more than 1 returner
punts_2020_no_onside = punts_2020.dropna(subset=['returnerId'])
remove_double_returners = lambda play: len(play['returnerId'].split(';')) == 1
punts_2020_no_onside['doubleReturners'] = punts_2020_no_onside.apply(remove_double_returners, axis='columns')
# before we group, get rid of rows with double returners
punts_2020_no_onside = punts_2020_no_onside[punts_2020_no_onside['doubleReturners'] == True]
grouped_kickoffs_no_onside = punts_2020_no_onside.groupby(punts_2020_no_onside['gameId'])
grouped_kickoffs_2_no_onside = punts_2020_no_onside.groupby([punts_2020_no_onside['gameId'], punts_2020_no_onside['playId']])
group_keys_no_onside = grouped_kickoffs_2_no_onside.groups.keys()

# loop through each group
for i in range(0, len(group_keys_no_onside)):
    group_game = list(group_keys_no_onside)[i][0]
    group_play = list(group_keys_no_onside)[i][1]
    # group dataframe from the original dataframe
    group = punts_2020[(punts_2020['gameId'] == group_game) & (punts_2020['playId'] == group_play)]
    # for this group, find the distance for each player
    group_returner = group['returnerId'].to_numpy()[0]
    returner_xy = group[group['nflId'] == int(group_returner)].loc[:, ['x', 'y']]
    f_eucl = lambda player: euclidean_distance((returner_xy['x'], returner_xy['y']), (player['x'], player['y']))
    group['distanceFromReturner'] = group.apply(f_eucl, axis='columns')
    new_punts_dataframe = new_punts_dataframe.append(group, ignore_index=True)


# remove rows where value is 0 on distance from returner because that would give us the closest teammate of the returner
new_punts_dataframe = new_punts_dataframe[new_punts_dataframe['distanceFromReturner'] != 0]
# group dataset by game, play, team
grouped = new_punts_dataframe.groupby(['gameId', 'playId', 'team'])
grouped_agg = new_punts_dataframe['distanceFromReturner'].groupby([new_punts_dataframe['gameId'], new_punts_dataframe['playId'], new_punts_dataframe['team']]).agg(['mean', 'std', 'min'])
# away is first, home is second
# create two data frames then ill merge them together by the keys
kicking_team_aggr = pd.DataFrame(columns=['kicking_team__mean', 'kicking_team_std', 'kicking_team_min'])
return_team_aggr = pd.DataFrame(columns=['return_team__mean', 'return_team_std', 'return_team_min'])
# for each row in the grouped aggregated dataframe...
for i in range(0, len(grouped_agg)):
    keys = grouped_agg.index[i]
    gameId = keys[0]
    playId = keys[1]
    team = keys[2]
    row_mean = grouped_agg.iloc[i]['mean']
    row_std = grouped_agg.iloc[i]['std']
    row_min = grouped_agg.iloc[i]['min']
    # find the play by game id and punt id
    punt_play = new_punts_dataframe[(new_punts_dataframe['gameId'] == gameId) & (new_punts_dataframe['playId'] == playId)]
    home_team = punt_play['homeTeamAbbr'].to_numpy()[0]
    away_team = punt_play['visitorTeamAbbr'].to_numpy()[0]
    kicking_team = punt_play['possessionTeam'].to_numpy()[0]
    # purpose of doing this is so that i can identify kicking team instead of home/away in the df
    # if the kicking team (possessionTeam) is the home team, then the 'kicking_team' is the home team
    kicking = ''
    # for this particular play...
    if kicking_team == home_team:
        kicking = 'home'
    else:
        kicking = 'away'

    # if the team is equal to the kicking variable then obviously this is the team thats kicking on this play
    if team == kicking:
        kicking_team_dict = {
            'gameId': gameId,
            'playId': playId,
            'kicking_team__mean': row_mean, 
            'kicking_team_std': row_std,
            'kicking_team_min': row_min
        }
        kicking_team_aggr = kicking_team_aggr.append(kicking_team_dict, ignore_index=True)
    else:
        return_team_dict = {
            'gameId': gameId,
            'playId': playId,
            'return_team__mean': row_mean, 
            'return_team_std': row_std,
            'return_team_min': row_min
        }
        return_team_aggr = return_team_aggr.append(return_team_dict, ignore_index=True)


both_aggr = pd.merge(kicking_team_aggr, return_team_aggr, on=['gameId', 'playId'])
# check how many games plays there are that were kickoffs and it was returned
# plays = pd.read_csv('../plays.csv')
right_df = pd.merge(plays, both_aggr, on=['gameId', 'playId'], how='right')
# merge the other dataframes
# games = pd.read_csv('../games.csv')
# pff = pd.read_csv('../pff.csv')
# continue merging based on the new dataframe with tracking data
right_df = pd.merge(games, right_df, on=['gameId'], how='right')
# continue merging based on the new dataframe with tracking data
right_df = pd.merge(pff, right_df, on=['gameId', 'playId'], how='right')
# new punt plays
# right_df.to_csv('punts_update2.csv', index=False)
# right_df.to_csv('punts_update_fixed_return_team_min.csv', index=False)


"""####################"""
# one hot encoding #
import seaborn as sb

# punts = pd.read_csv('../punts_update2.csv')
# punts = pd.read_csv('../punts_update_fixed_return_team_min.csv')
# columns i can drop, cant think of any other ones
punts_cols_to_drop = ['kickoffReturnFormation']
punts_one_hot_encoded = right_df.drop(punts_cols_to_drop, axis=1)
punts_dummy_columns = ['snapDetail', 'kickType', 'kickDirectionIntended', 'kickDirectionActual', 'returnDirectionIntended', 'returnDirectionActual', 'kickContactType']
for i in range(0, len(punts_dummy_columns)):
    column = punts_dummy_columns[i]
    dummies = pd.get_dummies(punts_one_hot_encoded[column], prefix=column)
    punts_one_hot_encoded = pd.concat([punts_one_hot_encoded, dummies], axis=1)
    punts_one_hot_encoded.drop([column], axis=1, inplace=True)


# correlations for dummy cols
# how about just the specific dummy cols created
snap_detail_dummy = ['snapDetail_<', 'snapDetail_>', 'snapDetail_H', 'snapDetail_L',
       'snapDetail_OK']
snap_detail = punts_one_hot_encoded.loc[:, snap_detail_dummy]
# plotting correlation heatmap
dataplot = sb.heatmap(snap_detail.corr(), cmap="YlGnBu", annot=True)
# displaying heatmap
plt.show()

kick_type_dummy = ['kickType_A', 'kickType_N', 'kickDirectionIntended_C',
       'kickDirectionIntended_L', 'kickDirectionIntended_R',
       'kickDirectionActual_C', 'kickDirectionActual_L',
       'kickDirectionActual_R',]

kick_type = punts_one_hot_encoded.loc[:, kick_type_dummy]
# plotting correlation heatmap
dataplot = sb.heatmap(kick_type.corr(), cmap="YlGnBu", annot=True)
# displaying heatmap
plt.show()

return_direction_dummy = ['returnDirectionIntended_C',
       'returnDirectionIntended_L', 'returnDirectionIntended_R',
       'returnDirectionActual_C', 'returnDirectionActual_L',
       'returnDirectionActual_R']
    
return_direction = punts_one_hot_encoded.loc[:, return_direction_dummy]
# plotting correlation heatmap
dataplot = sb.heatmap(return_direction.corr(), cmap="YlGnBu", annot=True)
# displaying heatmap
plt.show()


kick_contact_type_dummy = ['kickContactType_BC', 'kickContactType_BOG',
       'kickContactType_CC', 'kickContactType_CFFG', 'kickContactType_KTB',
       'kickContactType_MBDR']

kick_contact_type = punts_one_hot_encoded.loc[:, kick_contact_type_dummy]
# plotting correlation heatmap
dataplot = sb.heatmap(kick_contact_type.corr(), cmap="YlGnBu", annot=True)
# displaying heatmap
plt.show()