import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.optimize import newton


def system(t,v):
    x,y,z=v
    a=0.2
    b=0.2
    c=5.7
    dxdt=-y-z
    dydt=x+(a*y)
    dzdt=b+z*(x-c)
    return [dxdt,dydt,dzdt]

def P(a,x):
    return np.polyval(a,x)

def Poly_prime(a,x):
  return np.polyval(np.polyder(a), x)


def Q(a,x):
    return np.polyval(a,x)

def res(a,x_data,y_data):
    return P(a,x_data)-y_data

def Q_k(b, x, k):
    val=x
    for i in range(k):
        val = Q(b,val)
    return val

def Q_k_prime(b, x, k):
    val=x
    deriv=1
    for j in range(k):
        deriv*=Poly_prime(b, val) 
        val=Q(b, val)
    return deriv





t_span=(0,1500)
t_eval=np.linspace(0,1500,20000)
x0=[0,-1,0]

sol=solve_ivp(system,y0=x0,t_span=t_span,t_eval=t_eval,method='RK45')

fig=plt.figure()
ax=fig.add_subplot(projection='3d')

x_t=sol.y[0]
y_t=sol.y[1]
z_t=sol.y[2]

#grabbing the intersection points
x_vals=[]
z_vals=[]

for k in range(len(y_t)-1):
    if y_t[k]>0 and y_t[k+1]<=0: #if there exists a sign change in y
        diff=y_t[k+1]-y_t[k]
        if diff==0:
            continue
        step=-y_t[k]/diff
        x_n=x_t[k]+step*(x_t[k+1]-x_t[k])
        z_n=z_t[k]+step*(z_t[k+1]-z_t[k])
        if x_n<0:
            x_vals.append(x_n)
            z_vals.append(z_n)

x_points=np.array(x_vals)
z_points=np.array(z_vals)
y_points=np.zeros_like(x_points)

degree_poly_P=2
degree_poly_Q=4
a_vals=np.zeros(degree_poly_P+1)
b_vals=np.zeros(degree_poly_Q+1)
X=np.linspace(-9,-2,1000)

result_P=least_squares(res,a_vals,args=(x_points[20:],z_points[20:]))
a_fitted=result_P.x

result_Q=least_squares(res,b_vals,args=(x_points[20:-1],x_points[21:]))
b_fitted=result_Q.x

P_fit=P(a_fitted,X)
Q_fit=Q(b_fitted,X)

z_fill=[]
x_fill=[]
for j in range(1,4):
    P1_newton = newton(lambda x: Q_k(b_fitted, x,j)-x, -2, fprime=lambda x: Q_k_prime(b_fitted,x,j)-1)
    z_fill.append(P(a_fitted,P1_newton))
    x_fill.append(P1_newton)
    print(f"Periodic orbit of period {j}: x={P1_newton}, z={P(a_fitted,P1_newton)}")

z_fill=np.array(z_fill)
x_fill=np.array(x_fill)
y_fill=np.zeros_like(x_fill)





ax.plot3D(x_t,y_t,z_t,color='purple',lw=.8,alpha=.7)
ax.scatter3D(x_points,y_points,z_points,color='red',s=8,depthshade=False,label='Plane Intersections')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Rössler Attractor')
plt.show()

plt.figure(figsize=(6,5))
plt.scatter(x_points[20:],z_points[20:],color='blue',s=8,alpha=0.7)
plt.plot(X,P_fit,label="P(x)",linestyle="--",linewidth=2,color='red')
plt.xlabel(r'$x_n$',fontsize=12)
plt.ylabel(r'$z_n$',fontsize=12)
plt.title(r'Poincaré Section $(x, z)$ ')
plt.grid(True,linestyle='--',alpha=0.5)
plt.legend()
plt.show()

plt.figure(figsize=(6,5))
plt.scatter(x_points[20:-1],x_points[21:],color='blue',s=8,alpha=0.5)
plt.plot(X,Q_fit,label="Q(x)",linestyle="--",linewidth=2,color='red')
plt.xlabel(r'$x_n$',fontsize=12)
plt.ylabel(r'$x_{n+1}$',fontsize=12)
plt.title(r'Poincaré Map: $(x_{n},x_{n+1})$')
plt.legend()
plt.grid(True,linestyle='--',alpha=0.5)
plt.show()


plt.figure(figsize=(6,5))
plt.scatter(x_points[20:-1],x_points[21:],color='blue',s=8,alpha=0.5)
plt.plot(X,Q_fit,label="Q(x)",linestyle="--",linewidth=2,color='red')
plt.xlabel(r'$x_n$',fontsize=12)
plt.ylabel(r'$x_{n+1}$',fontsize=12)
plt.title(r'Poincaré Map: $(x_{n},x_{n+1})$')
plt.legend()
plt.grid(True,linestyle='--',alpha=0.5)
plt.show()


fig2=plt.figure()
ax2 = fig2.add_subplot(projection='3d')
ax2.plot3D(x_t,y_t,z_t,color='blue',lw=.8,alpha=.7)
ax2.plot3D(x_fill,y_fill,z_fill,'ro',label='Periodic Orbits')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
plt.title('Periodic orbits')
plt.show()