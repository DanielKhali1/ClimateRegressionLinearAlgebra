from numpy import *
import pandas as pd
from matplotlib.pyplot import *
from Functions import *
import warnings
from mpl_toolkits.mplot3d import Axes3D
warnings.filterwarnings("ignore")

dat2006 = array(pd.read_csv('dat_2006.csv',header=0))
#print(dat2006)

fig = figure()
ax = Axes3D(fig)
dat2006 = transpose(dat2006)

# Plot the surface.
ax.scatter(dat2006[0], dat2006[1], dat2006[2])

ax.set_xlim3d(-20000,20000)
ax.set_ylim3d(-20000,20000)
ax.set_zlim3d(-10,10)

ax.invert_xaxis()

show()