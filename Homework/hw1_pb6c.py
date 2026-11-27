import numpy as np 
import matplotlib.pyplot as plt
from scipy import optimize
import matplotlib.lines as mlines

def dx_dt(x,y): #dxdt
    return (-y+x*(1-x**2-y**2))

def dy_dt(x,y): #dydt
    return (x+y*(1-x**2-y**2))

 #vectors from -10 to 10 of 200 points
x_vals=np.linspace(-10,10,200)
y_vals=np.linspace(-10,10,200)

X,Y=np.meshgrid(x_vals,y_vals)  #creatin a grid for x and y points

plt.contour(X,Y,dx_dt(X,Y),levels=[0],colors='black',label='dx/dt=0')  #passing in the grid and dxdt
plt.contour(X,Y,dy_dt(X,Y),levels=[0],colors='red',label='dy/dt=0') #passing in the grid and dydt
plt.contour(X,Y,dx_dt(X,Y),colors='green',alpha=0.4,label='dx/dt')  #phase space
plt.contour(X,Y,dy_dt(X,Y),colors='blue',alpha=0.4,label='dy/dt')  #phase portrait of the system
plt.legend( loc='upper right') #legend
plt.show()