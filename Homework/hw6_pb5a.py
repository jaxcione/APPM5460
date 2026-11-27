import numpy as np
import matplotlib.pyplot as plt

def iteration(x_init,N):
    a=2.4
    b=1.4
    c=.8
    d=.5

    xn=np.empty(N+1)
    yn=np.empty(N+1)
    xn[0]=x_init[0]
    yn[0]=x_init[1]
   
    for k in range(1,N+1):
        xn[k]=np.sin(b*yn[k-1])+c*np.sin(b*xn[k-1])
        yn[k]=np.sin(a*xn[k-1])+d*np.sin(a*yn[k-1])


    

    return xn,yn


def box(x,y):
    x_min=np.min(x)
    y_min=np.min(y)

    x_max=np.max(x)
    y_max=np.max(y)

    points=np.column_stack((x,y)) #reorganizing to be in [(x,y)] form
    #returns [min_x,min_y] and [max_x,max{y}]
    min_val=np.array([x_min,y_min])
    max_val=np.array([x_max,y_max])

    square=(max_val-min_val).max() #bounding the attractor 

    normalized=(points-min_val)/square #normalizes into [0,1]x[0,1]

    bins=np.array([9000]) #constructing grid

    eps=1/bins #side length of a grid box
    counter=[]

    for n in bins:
        e=[np.linspace(0,1,n+1)for _ in range(points.shape[1])]
        H=np.histogramdd(normalized,bins=e)[0] #counts how many points land in each vox
        counter.append(np.sum(H>0)) #we count if a point exists in that grid
    counter=np.array(counter)

    return np.log(counter)/np.log(1/eps) #returning it


start=np.array([.7,.4])
N=int(1E7)
X,Y=iteration(start,N)

print(box(X,Y))

# X=X[50:]
# Y=Y[50:]
    
# plt.scatter(X,Y,s=0.1, alpha=0.2,label="(xn,yn) plotted",color="purple")
# plt.legend()
# plt.xlabel('xn')
# plt.ylabel('yn')
# plt.show()

