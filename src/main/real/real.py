import pandas as pd
import numpy as np
from func import show_rankings_1
file_path = "src/Dataset/yow_userstudy_raw.csv"
df = pd.read_csv(file_path)

df = df.dropna()
filtered_df = df[(df['RSS_ID'] == 14) | (df['RSS_ID'] == 10)]
filtered_df = filtered_df[filtered_df['classes'].str.contains('people', case=False) | filtered_df['classes'].str.contains('People')]
filtered_df = filtered_df[(filtered_df['user_id'] == 56)]
filtered_df.loc[filtered_df['RSS_ID'] == 14, 'RSS_ID'] = 0
filtered_df.loc[filtered_df['RSS_ID'] == 10, 'RSS_ID'] = 1
filtered_df = filtered_df[filtered_df['relevant'] != -1]
filtered_df['relevant']/= 5

filtered_df = filtered_df[['relevant', 'RSS_ID']].sort_values(by='RSS_ID')

sampled_df = filtered_df.sample(10).sort_values(by='RSS_ID')


sampled_df.rename(columns={'relevant': 'U','RSS_ID':'G'}, inplace=True)

noise = np.abs(np.random.normal(0.05, 1, len(sampled_df)))
sampled_df['U']+=noise

exp_df = show_rankings_1(sampled_df['U'],sampled_df['G'],2,5)