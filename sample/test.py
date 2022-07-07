import pandas as pd
import numpy as np

x = pd.Series(np.arange(14), index=np.arange(14))
y = pd.Series(np.arange(14), index=np.arange(14))

show_treated = pd.Series([1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0])

t_1s_idxs, = np.nonzero(show_treated.array)
t_0s_idxs, = np.nonzero(1 - show_treated.array)
actual_t_1 = x[t_1s_idxs]
actual_t_0 = x[t_0s_idxs]
preds_t_1 = y[t_1s_idxs]
preds_t_0 = y[t_0s_idxs]

print(x)
print(actual_t_1)
print(actual_t_0)
