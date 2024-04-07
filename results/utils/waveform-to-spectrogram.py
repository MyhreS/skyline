import pandas as pd

PATH_TO_INPUT_DATA = "/Users/simonmyhre/workdir/gitdir/sqml/projects/sm_multiclass_masters_project/pull_data/cache/datav3"
df = pd.read_csv(PATH_TO_INPUT_DATA + "/data.csv")

print(df.head(5))

