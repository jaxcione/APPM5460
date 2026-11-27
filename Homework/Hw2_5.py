import numpy as np
import matplotlib.pyplot as plt 


alpha=np.linspace(.1,1,5)


def W(x,N,alph):
    sum=np.zeros_like(x)
    
    for j in range(0,N+1):
        sum+= (2 ** (-j * alph)) * np.cos((2**j) * x)
    
    return sum

X=np.linspace(-15,15,1000)
store=[]
for m in alpha:
    y=W(X,20,m)
    plt.plot(X,y,label=f'Alpha = {m}')
plt.legend()
plt.title('W(X)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()