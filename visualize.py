import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('erwin_stats.csv')
grouped = df.groupby('tree')
dynamic_group = grouped.get_group('dynamic')
dynamic_group = grouped.get_group('dynamic')
plt.figure(figsize=(8, 5))
plt.bar(dynamic_group['model_name'], dynamic_group['mean'], yerr=dynamic_group['std'], capsize=5)
plt.xlabel('Model Name')
plt.ylabel('Mean Invariance Error')
plt.title('Dynamic Tree - Mean with Std Dev')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('dynamic_group_barchart.png')