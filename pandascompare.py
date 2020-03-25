import pandas as pd 

df1 = pd.read_csv("E:\AI-ML\datasets\\testdata.csv")
df2 = pd.read_csv("E:\AI-ML\datasets\\testdata1.csv")

# Sort data frames by columns
df1=df1.sort_values('last_name')
df2=df2.sort_values('last_name')

def pair_columns(df, col1, col2):
   return df[col1] + df[col2]

def paired_mask(df_1, df_2, col1, col2):
   return pair_columns(df_1, col1, col2).isin(pair_columns(df_2, col1, col2))

# Filter dataframe 2 based on given columns from data frame 1
df2= df2.loc[paired_mask(df2, df1, "first_name", "last_name")]
print("************************ DataFreme 2 \n",df2)
print("************************ DataFreme 1 \n",df1)

# Compare the two dataframes column wise and print differences
for column in df1.columns[1:]:
    print("*****Comparing column***: ",column)
    compare = df2[column].isin(df1[column]).value_counts(dropna=False)
    print(compare)
    print(df2[column].isin(df1[column]).value_counts(dropna=False))



# print(df1.count())
# print(len(df2))

# def dataframe_difference(df1, df2, which=None):
#     """Find rows which are different between two DataFrames."""
#     comparison_df = df1.merge(df2,
#                               indicator=True,
#                               how='outer')
#     if which is None:
#         diff_df = comparison_df[comparison_df['_merge'] != 'both']
#     else:
#         diff_df = comparison_df[comparison_df['_merge'] == which]
#     diff_df.to_csv('data/diff.csv')
#     return diff_df

# dataframe_difference(df1, df2,'right_only')

# def get_different_rows(source_df, new_df):
#     """Returns just the rows from the new dataframe that differ from the source dataframe"""
#     merged_df = source_df.merge(new_df, indicator=True, how='outer')
#     changed_rows_df = merged_df[merged_df['_merge'] == 'right_only']
#     return changed_rows_df.drop('_merge', axis=1)

# print("**************** get Different rows***********")
# diff_df=get_different_rows(df1, df2)
# print (diff_df)

# df_diff = pd.concat([df1,df2], sort=False).drop_duplicates(keep=False)
# print("**************** concat***********")
# print (diff_df)


# print("************************ Set",set(df1.columns).intersection(set(df2.columns)))

# print("************************ Set",df1['first_name'].isin(df2['first_name']).value_counts())

# #print(df1.count(axis='columns'))


# for column in df1.columns[1:]:
#     print("*****Inside for loop***: \n",column)
#     print("************************ Set\n",df1[column].isin(df2[column]).value_counts())



# print("************************ Sort\n",df3.sort_values('last_name'))
# df3=df3.sort_values('last_name')

# print("************************ final \n",df3)


