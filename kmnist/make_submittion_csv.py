import numpy as np
import pandas as pd
import sys

ans = np.loadtxt(sys.argv[1])
print(ans[:5])
ans = list(map(int, ans))

df = pd.DataFrame()
df["ImageId"] = range(1,10001)
df["Label"] = ans

print(df.head())
df.to_csv("submittion.csv", index=False)
