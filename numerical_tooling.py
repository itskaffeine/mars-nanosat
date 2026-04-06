import numpy as np
from scipy.differentiate import derivative
def matrix_derivative(f, x, shape, args=(), **kwargs):
    """Call SciPy's derivative() routine with a scalar-to-matrix function
    
    This function extend's SciPy's derivative() function to handle
    functions f: R -> R^mxn. The call signature is identical to that
    of derivative(), with the exception that following the evaluation point
    x, the shape of the output (m, n) must also be specified.

    Parameters
    ----------
    shape :  tuple of int
        Shape of output structure of f, in format (rows, columns).

    Notes
    -----
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.differentiate.derivative.html
    for full documenation of derivative() routine, including parameters and returns.

    :Authors:
        Erick White <erick.white@colorado.edu>
    """
    f_args = lambda x: f(x,*args)
    f_element = lambda x, i, j: f_args(x)[i,j]

    dfdx = np.empty((shape[0], shape[1]))

    for i in range(shape[0]):
        for j in range(shape[1]):
            dfdx[i, j] = derivative(np.vectorize(f_element, signature='(),(),()->()'), x, args=(i, j), **kwargs).df

    return dfdx