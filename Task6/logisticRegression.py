import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression

# <dataGeneration>
# make 2-class dataset consisting of two
# blobs, such that the dataset is not linearly separable.
centers = [[-5, -5], [0, 0], [5, 5]]
xy, clazz = make_blobs(n_samples=300, centers=centers, random_state=0)
clazz = [k % 2 for k in clazz] # this glues 0-th and 2-nd blob into one class
# </dataGeneration>

# <modifyThis>
def gaussianRbf(centerX, centerY, sigmaSquare, pointX, pointY):
  return 1 # HomeworkTodo: implement the gaussian RBF formula

def extractFeatures(xy):
  x = xy[0]
  y = xy[1]
  return [x, y] # HomeworkTodo: experiment with more interesting features
# </modifyThis>

# <training>
features = [extractFeatures(ab) for ab in xy]

clf = LogisticRegression(solver='sag', max_iter=10000, random_state=42).fit(
  features, # trains on the extracted features, not directly on coordinates!
  clazz
)
# </training>

# <plotting>
# create a mesh, classify each point in the mesh, color it accordingly
h = .02  # step size of the mesh
x_min, x_max = xy[:, 0].min() - 1, xy[:, 0].max() + 1
y_min, y_max = xy[:, 1].min() - 1, xy[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = clf.predict([extractFeatures(v) for v in np.c_[xx.ravel(), yy.ravel()]])
Z = Z.reshape(xx.shape)
plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.title("LogisticRegression with nonlinear features")
plt.axis('tight')

# Plot the training points
colors = "br"
for i, color in zip(clf.classes_, colors):
    idx = np.where(clazz == i)
    plt.scatter(xy[idx, 0], xy[idx, 1], c=color, cmap=plt.cm.Paired)

plt.show()
# </plotting>
