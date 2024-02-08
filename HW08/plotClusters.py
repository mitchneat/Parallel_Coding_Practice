import matplotlib.pyplot as plt
import numpy as np

rows = []
with open('clusterdata.txt','r') as reader:
    for line in reader:
        # Remove the newline character
        line = line.strip()
        # Make a list of the strings found between commas
        fields = line.split(' ')
        rows.append(fields)
rows.pop(0)

xTemp = []
yTemp = []
cTemp = []
for i in range(0, len(rows)):
    xTemp.append(rows[i][0])
    yTemp.append(rows[i][1])
    cTemp.append(rows[i][2])
x = np.array(xTemp).astype(float)
y = np.array(yTemp).astype(float)
c = np.array(cTemp).astype(float)

num_clusters = 18
my_colors = np.random.rand(num_clusters,3)

plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Clusters')


for i in range(0, num_clusters):
    for j in range(0, len(x)):
        if (i == c[j]):
            plt.plot(x[j], y[j], 'ro', color = my_colors[i])

plt.savefig('cluster.pdf')
plt.show()
