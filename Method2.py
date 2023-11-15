from scipy import optimize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

excel_path = [f"./STD/brownian-{i}.xlsx" for i in range(12, 21)]
save_path = './STD/STD_method2.png'

font = font_manager.FontProperties(family='Arial', size='16')

fig, ax = plt.subplots(figsize=(6, 5))
sum_msd = np.zeros(75, dtype=np.float64)
count_msd = 0


def read_data(path):
    df = pd.read_excel(path)
    df.columns = df.iloc[0]
    _, col = df.shape
    count = round((col + 1) / 4)

    for i in range(count):
        data = df.iloc[1:100, i * 4:i * 4 + 3]
        data = data - data.iloc[0]
        t = data.iloc[:75, 0]
        x = pd.to_numeric(data.iloc[:, 1], errors='coerce').dropna()
        # y = pd.to_numeric(data.iloc[:, 2], errors='coerce').dropna() # no use
        # msd(t) = sum(x(t + ti) - x(ti))**2 / N
        msd = pd.DataFrame(0, index=np.arange(75), columns=['msd'])
        global count_msd, sum_msd, ax
        for j in range(1, 75):
            tmpx = x.diff(j).dropna()
            tmp_msdx = np.mean(tmpx[:75] ** 2)
            msd.iloc[j] = tmp_msdx
            sum_msd[j] += tmp_msdx
        count_msd += 1
        ax.plot(t, msd, c='gray')


times = pd.read_excel(excel_path[0]).iloc[1:76, 0].to_numpy(dtype=np.float64)
for path in excel_path:
    try:
        read_data(path)
    except:
        print(f"error at {path}")
        continue

# linear fit
popt_linear, pcov_linear = optimize.curve_fit(
    lambda t, D: 2 * D * t,
    times,
    (sum_msd / count_msd))

# curve fit
popt_curve, pcov_curve = optimize.curve_fit(
    lambda t, D, v: 2 * D * t + v ** 2 * t ** 2,
    times,
    (sum_msd / count_msd))


def D_to_eta(D):
    return (1.380649E-23 * (26 + 273.15) / (6 * np.pi * (1.1e-6 / 2) * D))


# print
print(
    f"curve fit: D = {popt_curve[0]}, v = {popt_curve[1]}, eta = {D_to_eta(popt_curve[0])}")

# plot
ax.scatter(times, sum_msd / count_msd, c='k', label='data', s=2)
ax.plot(times, 2 * popt_linear[0] * times, c='r', label='linear fit')
ax.plot(times, 2 * popt_curve[0] * times + popt_curve[1]
        ** 2 * times ** 2, c='b', label='curve fit')
ax.set_xlabel('t (s)', fontsize=20, fontproperties='Arial', fontweight='bold')
ax.set_ylabel('MSD (m²)', fontsize=20,
              fontproperties='Arial', fontweight='bold')
ax.tick_params(which='both', direction='in', top=True, right=True)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
plt.tight_layout()
plt.legend(prop=font)
plt.savefig(save_path, dpi=600)
plt.show()
