import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp

def system(t,w):
    x,y,z=w

    dxdt=y*z
    dydt=x-1.7*y
    dzdt=1-x*y

    return [dxdt,dydt,dzdt]

t_span=(0,1000)
x0=[1,1,1]
t_eval=np.linspace(0,1000,100000)

sol=solve_ivp(system,y0=x0,t_eval=t_eval,t_span=t_span)

x=sol.y[0]
y=sol.y[1]
z=sol.y[2]

fig=plt.figure()
ax=fig.add_subplot(projection="3d")
ax.plot3D(x,y,z,color='purple',lw=1.3,alpha=.7)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Attractor with a=0.59')
plt.show()