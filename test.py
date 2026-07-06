import iir_quic
import numpy as np

# generate sample from tridiagonal precision matrix
n = 500  # the dimension of S
m = 250  # the number of independently drawn samples

# generate a tridiagonal matrix
np.random.seed(1)
a = -0.5 * np.ones(n-1)
b = 1.25 * np.ones(n)
Sigma_Inv = np.diag(a,-1) + np.diag(b,0) + np.diag(a,1)

# generate the data
L = np.linalg.cholesky(Sigma_Inv)
Y = np.linalg.solve(L.T,np.random.randn(n,m))
S = np.cov(Y)

X, f_val_list, KKT_list, time_list = iir_quic.iir_quic(S, 0.1)

print(f"{'Pass the test' if np.isclose(f_val_list[-1], 616.3827024900871) else 'Failed the test'}")
