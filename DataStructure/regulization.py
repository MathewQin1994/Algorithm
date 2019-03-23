import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
np.random.seed(2)
x=np.array([-8,0,1,2,3,3.5,4,4.5,6,10,20])
x1=np.linspace(-8,20,100)
y=np.matrix(-x**2+10*x+20+10*np.random.randn(len(x))).T
X=np.vstack((x,x**2,x**3,x**4,x**5,x**6,x**7)).T
X1=np.vstack((x1,x1**2,x1**3,x1**4,x1**5,x1**6,x1**7)).T

X=ss.fit_transform(X)
X1=ss.transform(X1)
X=np.matrix(np.hstack((np.array([[1]*len(x)]).transpose(),X)))
X1=np.matrix(np.hstack((np.array([[1]*len(x1)]).transpose(),X1)))
w=(X.T*X+0*np.matrix(np.eye(8))).I*X.T*y
y_hat=X1*w
plt.figure(1)
plt.plot(x,y)
plt.plot(x1,y_hat)
plt.show()