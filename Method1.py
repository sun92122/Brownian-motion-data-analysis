import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

excel_path = [f"./STD/brownian-{i}.xlsx" for i in range(12, 21)]
save_path = './STD/STD_method1.png'

fig, ax = plt.subplots(figsize=(6, 5))
all_point = []


def read_data(path):
    df = pd.read_excel(path)
    df.columns = df.iloc[0]
    _, col = df.shape
    count = round((col + 1) / 4)

    for i in range(count):
        data = df.iloc[1:, i * 4:i * 4 + 3]
        t = pd.to_numeric(data.iloc[:, 0], errors='coerce').dropna()
        x = pd.to_numeric(data.iloc[:, 1], errors='coerce').dropna() / 10
        global all_point
        all_point += (x.diff(1).dropna() / t.diff(1).dropna()).tolist()


for path in excel_path:
    try:
        read_data(path)
    except:
        print(f"error at {path}")
        continue

all_point = np.array(all_point)
all_point_ = all_point[abs(all_point) < np.std(all_point) * 3]
n, bins, patches = ax.hist(
    all_point_, bins=200, density=True, histtype='stepfilled', alpha=0.5)
y = ((1/(np.sqrt(2*np.pi) * np.std(all_point))) *
     np.exp(-0.5 * (1 / np.std(all_point) * (bins - np.mean(all_point))) ** 2))
print(np.mean(all_point), np.std(all_point))
ax.plot(bins, y)
ax.set_xlabel('x displacement (m)', fontproperties='Arial',
              fontsize=20, fontweight='bold')
ax.set_ylabel('frequency', fontproperties='Arial',
              fontsize=20, fontweight='bold')
ax.tick_params(which='both', direction='in', top=True, right=True)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
plt.tight_layout()
plt.savefig(save_path, dpi=600)
plt.show()
D = (np.std(all_point) ** 2 + np.mean(all_point)) / 2


def D_to_eta(D): return (1.380649E-23 * (26 + 273.15) /
                         (6 * np.pi * (1.1e-6 / 2) * D))


print(f"D = {D}, eta = {D_to_eta(D)}")
