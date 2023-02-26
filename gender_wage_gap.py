# http://alemartinello.com/2019/11/29/How-NOT-to-calculate-gender-gaps-in-wages/
# Simulate
import numpy as np
import pandas as pd

data = {
        'ability': list(range(1,11))*2,
        'gender': ['male']*10 + ['female']*10, 
}

data = pd.DataFrame(data)
data

data.loc[:, 'manager'] = (data.loc[:, 'ability']>8) | ((data.loc[:, 'gender']=='male') & (data.loc[:, 'ability']>5))

data.loc[:, 'wages'] = data.loc[:, 'ability'] + data.loc[:, 'ability']*data.loc[:, 'manager']
data.loc[:, 'wages'] = data.loc[:, 'wages'] - 0.1*(data.loc[:, 'gender']=='female')*data.loc[:, 'wages']

data

honest_gg = data.groupby('gender').mean()
print(honest_gg['wages'], '\n')
print(' Gender gap: {}%'.format(
    np.round(100*(1- honest_gg['wages'][0]/honest_gg['wages'][1]), 2)
    ))

hownotto_gg = data.groupby(['manager', 'gender']).mean()
print(hownotto_gg['wages'], '\n')

print(' Gender gap, grunts:  {}%\n'.format(
    np.round(100*(1- hownotto_gg['wages'][0][0]/hownotto_gg['wages'][0][1]), 2)
    ),
    'Gender gap, managers: {}%\n'.format(
    np.round(100*(1- hownotto_gg['wages'][1][0]/hownotto_gg['wages'][1][1]), 2)
    )
     )

print(hownotto_gg['ability'], '\n')
