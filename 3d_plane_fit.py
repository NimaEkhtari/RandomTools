# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:16:31 2023

This code uses different methods to calculate the normal vector of a plane given
a bunch of 3D points and also calculates the RMSE of the misclosure error.

@author: nekhtari
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull


A = -0.02
B = 0.06
C = 0.97
D = 1.2
n = 70

''' Randomly creating n points in 3D space given parameters A, B, C, and D '''
p = np.random.rand(n, 2)
pz = (D - np.sum(p * np.repeat(np.array([[A, B]]), n, axis = 0), axis = 1)) / C
pz = pz + (0.15 * np.random.randn(n, ))
points = np.column_stack((p, pz))



''' Method 1, using the normal vector '''
cr = points - np.mean(points, axis=0)   # Centroid removed points
Cov = np.cov(cr, rowvar=False)
[val, vec] = np.linalg.eig(Cov)
normal= vec[:, np.argmin(val)]

d = np.zeros((n, 1))
for jj in range(n):
    d[jj] = np.dot(normal, points[jj])
    
print(np.std(d))


''' Method 2, using SVD '''
a, b, c = np.linalg.svd(points - np.mean(points, axis=0))
print(min(b) / np.sqrt(n))
Normal = c[np.argmin(b)]


''' MEthod 3, Least squares adjustment along Z axis '''
AA = np.array([points[:, 0].T, points[:, 1].T, np.ones(n).T]).T
L = -points[:, 2]
X = np.matmul(np.linalg.inv(np.matmul(AA.T, AA)), np.matmul(AA.T, L))

nv = np.array([X[0], X[1], 1])
nv = nv / np.linalg.norm(nv)

print(normal)
print(Normal)
print(nv)




ch = ConvexHull(points[:, 0:2])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# x, y, z = [0,1,1,0], [0,1,0,1], [15, 15, 15, 15]
# verts = [list(zip(x, y, z))]
# ax.add_collection(Poly3DCollection(verts))


Ax1 = ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, marker='o')


plt.title('3D points')
plt.show()