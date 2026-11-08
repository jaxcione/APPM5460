import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def system(t,z):
    x,y=z

    dxdt=(1-u*x)*x**(3/2)-x**(1/2)*y
    dydt=p*(x-1)*y

    return [dxdt,dydt]

u=1/4
p=12

t_span=(0,100)
t_eval=np.linspace(0,100,10000)

x0=[.8,1-1/16]

sol=solve_ivp(system,t_span,x0,t_eval=t_eval)

plt.plot(sol.t,sol.y[0],label="x(t)")
plt.plot(sol.t,sol.y[1],label="y(t)")
plt.xlabel("t")
plt.ylabel("State Variables")
plt.title("Solution of the System of ODEs")
plt.legend()
plt.show()

plt.figure(figsize=(6,6))
plt.plot(sol.y[0],sol.y[1],label="Phase Space")
plt.plot(sol.y[0][0],sol.y[1][0],"ro",label="Initial Condition")
plt.xlabel("x(t)")
plt.ylabel("y(t)")
plt.title("Phase Space Plot")
plt.legend()
plt.show()