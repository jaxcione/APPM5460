import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ---------------- Parameters ----------------
R=1.987  #R in cal/molK
f=0.5 #fraction given in paper

alpha_p,alpha_m,alpha_s=0.00048,0.001,0.001 #free volumne
Tgp,Tgm,Tgs=387.0,167.0,181.0 #glass transition

#Molar volumes (m^3/mol) 
MW_M,MW_S=0.1001,0.0881 # kg/mol molecular weight of monomer (MMA) and solvent (EA)
rho_M,rho_P,rho_S=940.0,1190.0,900.0  # kg/m^3
vM=MW_M/rho_M
vP=MW_M/rho_P          
vS=MW_S/rho_S
eps=vM-vP  #contraction coefficient


dH=-5.77e4 #J/mol a fixed constant. Maybe we can have it vary with T as well in the future 
cp=1900.0#J/(kg K)
rho_in=921.0 #kg/m^3
U=300.0 # W/(m^2 K)
A=6 #m^2
Tc=330.0 #K
T_in=320.0 #K

# Feed
phiM_in=0.5
M_in=phiM_in/vM
S_in=(1-phiM_in)/vS
I_in=20.0 #mol/m^3 (0.02 mol/L)

tau=3600.0 #s residence time
V0=1.0 #m^3
Q_in=V0/tau # m^3

#-----------------------------------------------------------------------------------------------------

def rate_constants(T,vf):
    
    k0_p=4.92E5*np.exp(-4353/(R*T))*1E-3 #converting to SI
    k0_t=9.8E7*np.exp(-701/(R*T))*1E-3 #convering to SI

    kp=k0_p*gp(T,vf)
    kt=k0_t*gt(T,vf)
    kd=1.69E14*np.exp(-30000/(R*T))

    return kd,kp,kt


#gel effect part---------------------------------------------------------------------
def free_volume(M, S, T):
    phi_m=vM*M
    phi_s=vS*S
    phi_p=1.0-phi_m-phi_s    
   
    Vfp=.025+alpha_p*max(T-Tgp, 0.0)
    Vfm=.025+alpha_m*max(T-Tgm, 0.0)
    Vfs=.025+alpha_s*max(T-Tgs, 0.0)
    return Vfp*phi_p+Vfm*phi_m+Vfs*phi_s #returning the free volume


def gt(T,vf):
    if vf>.1856-2.965E-4*(T-273.2):
       return .010575*np.exp(17.15*vf-.001715*(T-273.2))
    else:
        return 2.3E-6*np.exp(75*vf)
    
def gp(T,vf):
    if vf>.05:
       return 1
    else:
        return 7.1E-5*np.exp(171.53*vf)

#---------------------------------------------------------------------

def system(t,w):
 
    T,p,M,I,S,VR=w
    VF=free_volume(M,S,T)

    #kinetic constants
    kd,kp,kt=rate_constants(T,VF)

    
    RM=(kp*(f*kd*I/kt)**0.5)*M
    Qout=Q_in-RM*eps*VR
    dVRdt = Q_in - Qout - eps*RM*VR  #should be zero 
    
    dMdt=-RM
    dIdt=((Q_in*I_in-Qout*I)/VR)-kd*I
    dSdt=1/VR*(Q_in-Qout)*S
    dpdt=1/VR*(rho_in*Q_in-p*Qout)
    dTdt=(1.0/(VR*p))*(T_in*rho_in*Q_in-T*Qout*p+(-dH*RM*VR+U*A*(Tc-T))/cp)-(T/p)*dpdt

    return [dTdt,dpdt,dMdt,dIdt,dSdt,dVRdt]

x0=[T_in,rho_in,M_in,I_in,S_in,V0]

t_span=(0,36000) #10 hours
t_eval=np.linspace(0,36000,100000)
sol=solve_ivp(system,y0=x0,t_eval=t_eval,t_span=t_span,method="BDF",rtol=1E-8)

T1,p1,M1,I1,S1,V1=sol.y
conversion=1-(M1/M_in)


plt.plot(t_eval, conversion,label=r"Conversion of MMA versus residence time $\tau = \dot{Q}_{in}/V_R$")
plt.xlabel("Residence Time")
plt.ylabel("Conversion of MMA")
plt.legend()
plt.show()