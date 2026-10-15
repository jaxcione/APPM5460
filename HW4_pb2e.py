import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def system(t,q): #our system of equations
    x,y,z=q

    dxdt=y
    dydt=np.sin(x)*(z**2*np.cos(x)-1)-e*y
    dzdt=a*np.cos(x)-a*b

    return [dxdt,dydt,dzdt]

#our constants which are set arbitrarily atm 
a=1
b=1/2
e=2*a*b**(3/2)-.01

#initial starting vector aat our eq point

q0 = [np.arccos(b) + 0.05, 0.02, 1 / np.sqrt(b) - 0.05]
t_span=(0,100)# going from -20 to 20
t_eval=np.linspace(0,100,1000)


sol=solve_ivp(system,t_span,q0,t_eval=t_eval) #solving the system
plt.plot(sol.t,sol.y[0],label="x(t)") #plotting xt
plt.plot(sol.t,sol.y[1],label="y(t)") #plotting yt
plt.plot(sol.t,sol.y[2],label="z(t)") #plotting zt

plt.xlabel('t')
plt.legend()
plt.grid()
plt.show()
