import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame.from_dict({
    'GPU_Name': ['4070', '3060', '3080', '4070 TI'],
    'Memory': [12, 12, 12, 12],
    'Cuda_Cores': [5888, 3584, 8960, 7680],
    'Price': [600, 300, 600, 755]
})

sns.lineplot(data=df, x='Cuda_Cores', y='Price')

plt.show()