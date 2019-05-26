import numpy as np 


def laplacian_spectrum(L):
    eigenvals, eigenvecs = np.linalg.eigh(L)
    idx = eigenvals.argsort()
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:,idx]
    def transform(x, i=0, k=1):
        return eigenvecs[:,i:i+k].T.dot(x)
    def inv_transform(xh, i=0, k=1):
        return eigenvecs[:,i:i+k].dot(xh)
    return transform, inv_transform, eigenvals


def auglap_spectrum(L):
    d = np.diagonal(L) # D
    d = d + 1  # \tilde{D}
    d_inv_sqrt = np.power(d, -1/2) # \tilde{D}^{-1/2}
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_inv_sqrt = np.diag(d_inv_sqrt)
    eigenvals, eigenvecs = np.linalg.eigh(d_inv_sqrt.dot(L).dot(d_inv_sqrt))
    idx = eigenvals.argsort()
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:,idx]
    d_sqrt = np.diag(np.power(d, 1/2)) # \tilde{D}^{-1/2}
    def transform(x, i=0, k=1):
        return eigenvecs[:,i:i+k].T.dot(d_sqrt).dot(x)
    def inv_transform(xh, i=0, k=1):
        return d_inv_sqrt.dot(eigenvecs[:,i:i+k]).dot(xh)
    return transform, inv_transform, eigenvals


# Monkey patch incase we need pure features
def identity(L):
    return lambda x: x, lambda x: x


def fetch_transformation(name):
    switcher = {
        '': identity,
        'GFT': laplacian_spectrum,
        'NGFT': auglap_spectrum
    }
    return switcher[name]