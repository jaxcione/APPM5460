import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp



def system(t,w,a):
    x,y,z=w

    dxdt=y*z
    dydt=x-a*y
    dzdt=1-x*y

    return [dxdt,dydt,dzdt]

def Jacobian(w,a):
    x,y,z=w

    A=np.array([[0,z,y],[1,-a,0],[-y,-x,0]])
    return A


def repack(t,s,a):
    x=s[:3]
    Q=s[3:].reshape(3,3)

    return np.concatenate([system(t,x,a),(Jacobian(x,a)@Q).ravel()]) #.ravel flattens an array

    
def Ly_spectrum(x0,N,discard,a):
    
    Q0=np.eye(3)
    u=np.zeros(3)
    
    
    for j in range(discard+N):
        
            s0=np.concatenate([x0,Q0.ravel()]) #conjoinging xo Q0

            sol=solve_ivp(repack,(0,1),s0,args=(a,),rtol=1e-9,atol=1e-12,method='DOP853').y[:,-1] #solving Q'=JQ
            x0=sol[:3] #new point
            Zk=sol[3:].reshape(3,3)

            Q0,R=np.linalg.qr(Zk)

            if j>=discard: #only add after transient term
                u+=np.log(np.abs(np.diag(R)))
    

    return u/N


a=np.arange(.4,1.71,.01)
store_u=[]
x0=[1,1,1]
for k in a:
     store_u.append(Ly_spectrum(x0,100,5,k)[0])
store_u=np.array(store_u)

plt.plot(a,store_u,label="u_1(a) where a∈(.4,1.7)")
plt.title("u_1(a)")
plt.xlabel("a")
plt.ylabel("u_1(a)")
plt.legend()
plt.show()






