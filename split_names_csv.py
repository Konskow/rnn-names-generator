import pandas as pd
import numpy as np


def save_list(path, list):
    with open(path, 'w') as f:
        for name in np.sort(list):
            f.write(name + '\n')


if __name__ == '__main__':
    df = pd.read_csv('data/english/usa_names_agg.csv')

    female_df = df.loc[df['gender'] == 'F']
    female_df = female_df.loc[female_df['number'] > 500]
    male_df = df.loc[df['gender'] == 'M']
    male_df = male_df.loc[male_df['number'] > 500]

    female_names = np.unique(female_df['name'].values)
    male_names = np.unique(male_df['name'].values)

    save_list('data/english/female.txt', female_names)
    save_list('data/english/male.txt', male_names)
