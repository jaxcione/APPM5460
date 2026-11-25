import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp



def system(t,w):
    x,y,z=w

    dxdt=y*z
    dydt=x-y
    dzdt=1-x*y

    return np.array([dxdt,dydt,dzdt])

def Jacobian(w):
    x,y,z=w

    A=np.array([[0,z,y],[1,-1,0],[-y,-x,0]])
    return A



def Ly_spectrum(x0,N,discard):
    
    Q0=np.eye(3)
    u=np.zeros(3)
    
    for _ in range(discard):
        J0=Jacobian(x0)
        Zk=J0@Q0 
        z1=Zk[:,0]
        z2=Zk[:,1]
        z3=Zk[:,2]

        r11=np.linalg.norm(z1)
        q1=z1/r11

        r12=np.dot(q1,z2)

        w=z2-r12*q1
        r22=np.linalg.norm(w)
        q2=w/r22

        r13=np.dot(q1,z3)
        r23=np.dot(q2,z3)

        w2=z3-r13*q1-r23*q2
        r33=np.linalg.norm(w2)
        q3=w2/r33

        
        Q0=np.column_stack([q1,q2,q3])
        u+=np.log([r11,r22,r33])

    return u/N
        




