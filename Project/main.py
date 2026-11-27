import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def rate_constants(T):
    R=8.314 #J/molk
    k_d=1.69E14*np.exp(-30,000(R*T))
    k_p0=4.92E5*np.exp(-4353/(R*T))
    k0_t=9.8E7*np.exp(-701/(R*T))
    k_tc=.043


def gt(T,vf):
    if vf>.1856-2.965E-4*(T-273.2):
       return .010575*np.exp(17.15*vf-.001715*(T-273.2))
    else:
        return 2.3E-6*np.exp(75*vf)
    
def gp(T,vf):
    if vf>.005:
       return 1
    else:
        return 7.1E-5(171.53*vf)

def Vfp(T):


def system(t,w):
    #constants
    Qin=1 #L/s
    Qout=.4 #L/s
    Pin=2 #g/L ---> density
    eps=.3 #L/mol
    f=.5 #no unit
   

    
    


    T,p,M,I,VR=w

    #kinetic constants
    kd=k


    dMdt=((Qin*M-Qout*M)/VR)
    dVRdt=Qin-Qout-(dMdt*eps*VR)