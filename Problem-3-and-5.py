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


'''The discerete l_2-norm for a vector x'''
def d_norm(x):
    l=1/np.sqrt(len(x))
    return l*linalg.norm(x)

'''The continuous L_2-norm for a function (represented as a vector y)'''
def c_norm(x,y,a,b):
    X,Y=np.polynomial.legendre.leggauss(10)
    #shifting X:
    for i in range(10):
        X[i]=(b-a)/2*X[i]+(b+a)/2
    U=(CubicSpline(x,y)(X))**2
    I=(b-a)/2*(sum(Y[j]*U[j] for j in range(10)))
    return np.sqrt(I)



# ## Problem 3
# #### b)

# In[ ]:


def analytical(Mx,My):
    h=1/(Mx+1)
    k=1/(My+1)
    u=np.zeros(Mx*My)
    j=1
    l=1
    X=np.zeros((Mx*My,2))
    for i in range(Mx*My):
        x=l*h
        y=j*k
        X[i]=[x,y]
        u[i]=(1/(np.sinh(2*np.pi)))*np.sin(2*np.pi*x)*np.sinh(2*np.pi*y)
        if j%My==0:
            j=0
            l+=1
        j+=1
    return u,X



def Laplace(Mx,My):
    M=Mx*My
    f=np.zeros(Mx*My)
    h=1/(Mx+1)
    k=1/(My+1)
    A = np.diag(-4*np.ones(M)) + np.diag(np.ones(M-1),k=1) + np.diag(np.ones(M-My),k=My) + np.diag(np.ones(M-My),k=-My) + np.diag(np.ones(M-1),k=-1)
    i=1
    for j in range(M+1):
        if j%My==0 and j<M:
            A[j-1,j]=0
            A[j,j-1]=0
        if j%My==0 and j!=0:
            f[j-1]=np.sin(2*np.pi*i*h)
            i+=1
    A=-A
    U=linalg.solve(A,f)

    return U

Mx=3
My=5
U=Laplace(Mx,My)
u,X=analytical(Mx,My)


# In[ ]:


M=np.array([5,10,30,50,100])
error=np.zeros(len(M))
for i,m in enumerate(M):
    U=Laplace(m,m)
    u,X=analytical(m,m)
    error[i]=d_norm(u-U)/d_norm(u)

    
    
plt.plot(M**2,error)
plt.yscale("log")
plt.xscale("log")
plt.xlabel('Number of grid-points MxMy')
plt.ylabel('Relative error in discrete norm')
plt.grid()
plt.show()    


# ## Problem 5
# 

# In[ ]:


def FEM(X,a,b,g,d1,d2,deg=10):
    N=len(X)
    h=(b-a)/N
    
    A = 1/h*(np.diag((2)*np.ones(N)) + np.diag(-np.ones(N-1),k=-1) + np.diag(-np.ones(N-1),k=1))
    A[0,0]=1/h
    A[-1,-1]=1/h
    x,y=np.polynomial.legendre.leggauss(deg)
    f=np.zeros(N)
    for i in range(N-1):
        ai=X[i]
        bi=X[i+1]
        f[i]=(bi-ai)/2*(sum(y[j]*g((bi-ai)/2*x[j]+(ai+bi)/2) for j in range(deg)))
    u=np.zeros(N)
    u[0]=d1
    u[-1]=d2
    f=f-np.dot(A,u)
    A=A[1:-1,1:-1]
    A=csc_matrix(A)
    f=f[1:-1]
    U=spsolve(A,f)
    U=np.append(np.array(d1),U)
    U=np.append(U,d2)
    
    return X,U

deg=20
N_array=np.array([8, 16, 32, 64, 128, 256, 512, 1024, 2048])

def FEM_error_plot(N_array,a,b,f,d0,d1,u):
    error=np.zeros(len(N_array))
    for i,n in enumerate(N_array):
        X=np.linspace(a,b,n)
        X,U=FEM(X,a,b,f,d0,d1,deg)
        error[i]=c_norm(X,u(X)-U,a,b)/(c_norm(X,u(X),a,b))
    plt.plot(N_array,error) 
    plt.yscale("log")
    plt.xscale("log")
    plt.grid()
    plt.show() 


# In[ ]:


def error_norm(u,Uc,a,b):
    X=np.linspace(a,b,10)
    I=(u(X)-Uc(X))**2
    x,Y=np.polynomial.legendre.leggauss(10)
    #shifting X:
    for i in range(10):
        x[i]=(b-a)/2*x[i]+(b+a)/2
    I=(b-a)/2*(sum(Y[j]*I[j] for j in range(10)))
    return np.sqrt(I)
    

def AFEM(N,f,u,a,b,d1,d2,alpha,estimate='averaging'):
    X=np.linspace(a,b,N)
    X,U=FEM(X,a,b,f,d1,d2,deg=10)
    #plt.plot(X,U,color='green')
    errors=10*np.ones(len(X)-1)
    
    for j in range(5):
        if estimate=='averaging':
            '''Averaging error estimate:'''
            E=alpha*c_norm(X,u(X)-U,a,b)/N
        if estimate=='maximum':
            maximum=(np.absolute(u(X)-U)).max()
            E=alpha*maximum

        Uc=CubicSpline(X,U)
        
        '''Error on each grid-interval:'''
        for i in range(len(X)-1):
            errors[i]=error_norm(u,Uc,X[i],X[i+1])

        if np.sum(errors)<len(errors)*E:
            return X,U 
        
        '''Adding grid-points where it is necessary:'''
        k=0
        for i,e in enumerate(errors):
            if e>=E:
                X=np.insert(X,i+1+k,(X[i+1+k]+X[i+k])/2)
                k+=1
        '''Making sure that the first two grid elements are of same length'''
        if np.abs((X[1]-X[0])-(X[2]-X[1]))>=1e-5:
            X=np.insert(X,1,(X[1]+X[0])/2)
            
        '''Solving the system with respect to the new grid'''
        X,U=FEM_non_uniform(X,f,a,b,d1,d2)
        errors=np.ones(len(X)-1)
    return X,U
        
      
def FEM_non_uniform(X,g,alpha,beta,d1,d2):
    N=len(X)
    h=np.zeros(N-1)
    h2=np.zeros(N)
    '''The grid elements:'''
    for i in range(0,N-1):
        h[i]=1/(X[i+1]-X[i])
    for i in range(1,N-1):
        h2[i]=h[i]+h[i-1]
    h2[0]=h[0]
    h2[-1]=h[-1]
    A = np.diag(h2) + np.diag(-h,k=-1) + np.diag(-h,k=1)
    f=np.zeros(N)
    x,y=np.polynomial.legendre.leggauss(deg)
    for i in range(N-1):
        ai=X[i]
        bi=X[i+1]
        f[i]=(bi-ai)/2*(sum(y[j]*g((bi-ai)/2*x[j]+(ai+bi)/2) for j in range(deg)))
    u=np.zeros(N)
    u[0]=d1
    u[-1]=d2
    f=f-np.dot(A,u)
    A=A[1:-1,1:-1]
    A=csc_matrix(A)
    f=f[1:-1]
    U=spsolve(A,f)
    U=np.append(np.array(d1),U)
    U=np.append(U,d2)
    #plt.plot(X,U,color='aqua')
    return X,U


# #### b)

# In[ ]:


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

