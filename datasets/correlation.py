from matplotlib import pyplot as plt
import pandas as pd

df = pd.read_csv("twins/twins.csv")

corr_matrix = df.corr(
    method='pearson',  # The method of correlation
    min_periods=1      # Min number of observations required
)

fig, ax = plt.subplots(figsize=(16, 12))
im = ax.imshow(corr_matrix, cmap="RdYlBu")
fig.colorbar(im)
ax.set_xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=45)
ax.set_yticks(range(len(corr_matrix)), corr_matrix.index)

for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        text = ax.text(j, i, round(corr_matrix.iloc[i, j], 1),
                       ha="center", va="center", color="w")

fig.savefig("twins-corr-matrix.png")
fig.show()

print(df.head())
