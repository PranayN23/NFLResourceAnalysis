import pandas as pd

#reads in the files from within the computer
rb_18_df = pd.read_csv("PFF/RB2018.csv")
rb_19_df = pd.read_csv("PFF/RB2019.csv")
rb_20_df = pd.read_csv("PFF/RB2020.csv")
rb_21_df = pd.read_csv("PFF/RB2021.csv")
rb_22_df = pd.read_csv("PFF/RB2022.csv")

#check your columns to make sure they are consistent and share the same names/shape
print("RB2018 columns:", rb_18_df.columns)
print("RB2019 columns:", rb_19_df.columns)
print("RB2020 columns:", rb_20_df.columns)
print("RB2021 columns:", rb_21_df.columns)
print("RB2022 columns:", rb_22_df.columns)
print(rb_19_df.shape)

#using outer merge to combine each db 1 by 1 and renaming each new right joint db based on the year of db
rb_combined_df = pd.merge(rb_18_df, rb_19_df, on='player', how='outer', suffixes=('_2018', '_2019'), )
rb_combined_df = pd.merge(rb_combined_df, rb_20_df, on='player', how='outer', suffixes=('', '_2020'))
rb_combined_df = pd.merge(rb_combined_df, rb_21_df, on='player', how='outer', suffixes=('', '_2021'))
rb_combined_df = pd.merge(rb_combined_df, rb_22_df, on='player', how='outer', suffixes=('', '_2022'))

#creates csv with combined df
rb_combined_df.to_csv('RB19-22.csv',index=False)