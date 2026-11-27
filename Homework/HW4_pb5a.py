import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def system_driver(t, w):
    xD,yD,zD,xR,yR,zR = w

    dxDdt=-yD- zD
    dyDdt=xD+a*yD
    dzDdt=b+zD*(xD-c)

    ex=xR-xD
    ey=yR-yD
    ez=zR-zD
    
    dxRdt=-yR-zR+(-zD*ez+ez-1.0*ex)
    dyRdt=xR+a*yR+(-2.0*a*(yR-yD)-1.0*(yR-yD)) # includes your -2a(yR-yD) term + damping
    dzRdt=b+zR*(xR-c)+(-xR*ez-1.0*ez)

    return [dxDdt,dyDdt,dzDdt,dxRdt,dyRdt,dzRdt]

a=.2
b=.2
c=5.7

tspan=(0,100)
t_eval=np.linspace(0,100,1000)
X0=[0,0,0,-5,3,15] 

sol=solve_ivp(system_driver,tspan,X0,t_eval=t_eval)
fig=plt.figure() #empty figure 

xD,yD,zD=sol.y[0],sol.y[1],sol.y[2]
xR,yR,zR=sol.y[3],sol.y[4],sol.y[5]

ax=fig.add_subplot(111,projection="3d") #object of fig in which things are plotted to. These are the axes
ax.plot(xD,yD,zD,color='red',label="xD,yD,zD",alpha=1) #plotting x,y,z solutions oon the axes
ax.plot(xR,yR,zR,color='black',label="xR,yR,zR",alpha=.9,linestyle="--")


#labeling them 
ax.set_xlabel('X') 
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()
