import pandas as pd
df_test = pd.read_csv('df_tr.csv', index_col=0)
print(list(df_test.columns))