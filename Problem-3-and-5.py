#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from scipy import integrate
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicSpline
from scipy import linalg




# #### b)

# In[ ]:
N_array=np.array([8, 16, 32, 64, 128, 256, 512, 1024, 2048])


def f(x):
    return -2

def u(x):
    return x**2

a=0
b=1
d1=0
d2=1

FEM_error_plot(N_array,a,b,f,d1,d2,u)

N=20
alpha1=1
estimate1='averaging'
alpha2=0.7
estimate2='maximum'

X1,U1=AFEM(N,f,u,a,b,d1,d2,alpha1,estimate1)
plt.plot(X1,u(X1),'.',color='hotpink')
plt.plot(X1,U1,color='aqua')
plt.grid()
plt.title(len(X1))
plt.show()
X2,U2=AFEM(N,f,u,a,b,d1,d2,alpha2,estimate2)
plt.plot(X2,u(X2),'.',color='hotpink')
plt.plot(X2,U2,color='aqua')
plt.title(len(X2))
plt.grid()


# #### c)

# In[ ]:


def f(x):
    return -(40000*x**2-200)*np.exp(-100*x**2)

def u(x):
    return np.exp(-100*x**2)

a=-1
b=1
d1=np.exp(-100)
d2=np.exp(-100)

FEM_error_plot(N_array,a,b,f,d1,d2,u)


N=20
alpha1=1
estimate1='averaging'
alpha2=0.7
estimate2='maximum'

X1,U1=AFEM(N,f,u,a,b,d1,d2,alpha1,estimate1)
plt.plot(X1,u(X1),'.',color='hotpink')
plt.plot(X1,U1,color='aqua')
plt.grid()
plt.title(len(X1))
plt.show()
X2,U2=AFEM(N,f,u,a,b,d1,d2,alpha2,estimate2)
plt.plot(X2,u(X2),'.',color='hotpink')
plt.plot(X2,U2,color='aqua')
plt.title(len(X2))
plt.grid()


# #### d)

# In[ ]:


def f(x):
    return -(4000000*x**2-2000)*np.exp(-1000*x**2)

def u(x):
    return np.exp(-1000*x**2)

a=-1
b=1
d1=np.exp(-1000)
d2=np.exp(-1000)

FEM_error_plot(N_array,a,b,f,d1,d2,u)

N=20
alpha1=1
estimate1='averaging'
alpha2=0.7
estimate2='maximum'

X1,U1=AFEM(N,f,u,a,b,d1,d2,alpha1,estimate1)
plt.plot(X1,u(X1),'.',color='hotpink')
plt.plot(X1,U1,color='aqua')
plt.grid()
plt.title(len(X1))
plt.show()
X2,U2=AFEM(N,f,u,a,b,d1,d2,alpha2,estimate2)
plt.plot(X2,u(X2),'.',color='hotpink')
plt.plot(X2,U2,color='aqua')
plt.title(len(X2))
plt.grid()


# #### e)

# In[ ]:


def f(x):
    return 2/9*x**(-4/3)


def u(x):
    return x**(2/3)

a=0
b=1
d1=0
d2=1

FEM_error_plot(N_array,a,b,f,d1,d2,u)

N=20
alpha1=1
estimate1='averaging'
alpha2=0.7
estimate2='maximum'

X1,U1=AFEM(N,f,u,a,b,d1,d2,alpha1,estimate1)
plt.plot(X1,u(X1),'.',color='hotpink')
plt.plot(X1,U1,color='aqua')
plt.grid()
plt.title(len(X1))
plt.show()
X2,U2=AFEM(N,f,u,a,b,d1,d2,alpha2,estimate2)
plt.plot(X2,u(X2),'.',color='hotpink')
plt.plot(X2,U2,color='aqua')
plt.title(len(X2))
plt.grid()

