import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# bmh
plt.style.use('bmh')

import numpy as np

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Verdana'

matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['axes.titlesize'] = 8
matplotlib.rcParams['axes.labelsize'] = 9.5
matplotlib.rcParams['xtick.labelsize'] = 9.5
matplotlib.rcParams['xtick.color'] = 'Black'
matplotlib.rcParams['ytick.labelsize'] = 9.5
matplotlib.rcParams['ytick.color'] = 'Grey'
matplotlib.rcParams['legend.fontsize'] = 8
matplotlib.rcParams['hatch.color'] = 'Red'
matplotlib.rcParams['hatch.linewidth'] = 1.5
#matplotlib.rcParams['legend.edgecolor'] = 'Black'

hatches = ('', '///', '++', '\\\\\\', '//')
linestyles = ['-', '-', '-', '-', '-', '-', '-']
# markers = ['.', ',', 'o', 'v', '^', '<', '>', '1',
#            '2', '3', '4', '8', 's', 'p', 'P', '*', 'h',
#            'H', '+', 'x', 'X', 'D', 'd', '|', '_', '$yolo $',
#            'None']
markers = ['', 'o', 'P', 'X', '^', 'd', 's']
marker_freq = 8
colors = ['Red', 'Skyblue', 'Orange', 'Grey', 'MediumSlateBlue', 'Tomato', 'LightGrey', 'Palegreen', 'Azure']

methods = ['AllReduce',
           'Fagin',
           'Cluster']

N = 3
ind = np.arange(N)
width = 0.2

# comp, comm, total
allreduce = [110000, 6185, 116185]
fagin = [126, 9, 135]
cluster = [26, 3.3, 29.3]


def autolabel(rects, labels):
    """Attach a text label above each bar in *rects*, displaying its height."""
    index = 0
    for rect in rects:
        height = rect.get_height()
        height = labels[index]
        plt.annotate(str(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
        index += 1


ax = plt.figure(figsize=(6, 2))

rect1 = plt.bar(ind, allreduce, width, color=colors[1], hatch=hatches[0], label=methods[0])
rect2 = plt.bar(ind+width, fagin, width, color=colors[2], hatch=hatches[1], label=methods[1])
rect3 = plt.bar(ind+2*width, cluster, width, color=colors[3], hatch=hatches[2], label=methods[2])

autolabel(rect1, allreduce)
autolabel(rect2, fagin)
autolabel(rect3, cluster)

plt.title("k=10K")
plt.ylabel("Run time (seconds)")
plt.ylim(1, 100000000)
#plt.yticks([0.5, 1, 10, 100, 1000], ['0.5', '1', '10', '100', '1000'])
plt.yscale('log')
plt.xticks(ind+0.2, ['Computation', 'Communication', 'Total'])
#plt.legend(bbox_to_anchor=(0, 1), loc=2, ncol=1, shadow=False)
plt.legend(framealpha=1, loc='upper left', ncol=4, shadow=False)

plt.tight_layout()

plt.savefig("time_breakdown_k10k.pdf")
plt.show()
