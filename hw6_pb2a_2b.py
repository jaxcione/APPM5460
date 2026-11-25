import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp

def system(t,w):
    x,y,z=w

    dxdt=y*z
    dydt=x-y
    dzdt=1-x*y

    return [dxdt,dydt,dzdt]

t_span=(0,1000)
x0=[1,1,1]
t_eval=np.linspace(0,1000,100000)

sol=solve_ivp(system,y0=x0,t_eval=t_eval,t_span=t_span)

x=sol.y[0]
y=sol.y[1]
z=sol.y[2]

new_x0=[1,1,1+10**-8]
sol2=solve_ivp(system,y0=new_x0,t_eval=t_eval,t_span=t_span)

x_1=sol2.y[0]
y_1=sol2.y[1]
z_1=sol2.y[2]

W=np.array([x_1,y_1,z_1])
B=np.array([x,y,z])

diff=np.linalg.norm((W-B),axis=0)
diff=np.log(np.abs(diff))

plt.plot(t_eval,diff,label="Log difference between x0 and x1",color='purple')
plt.xlabel("Time")
plt.ylabel('Ln(||X0-X1||)')
plt.show()

fig=plt.figure()
ax=fig.add_subplot(projection="3d")
ax.plot3D(x,y,z,color='green',lw=1.3,alpha=.7)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend(loc='upper left',bbox_to_anchor=(1.05, 1))
plt.title("System of ODEs")
plt.show()



