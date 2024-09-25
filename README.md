# torch-pchip_interp
Differentiable piecewise Hermite spline interpolation in pytorch

Simple standalone file that allows for interpolation of general pytorch tensors on irragular 1d and 2d grids. In 2d, the data must be on a regular grid, but it is not necessary to have equidistant grids or to be using the same number of grid points in the first and second spatial dimension.

### cubic_interp1d(x,y,xs): 
Takes as input: 

* x  -- input grid, shape (N,)
* y  -- input tensor y, shape (...,N)
* xs -- interp grid xs, shape (Ns,)

and outputs:

* ys -- interpolated tensor, shape (...,Ns)


### cubic_interp2d(x1,x2,y,xs1,xs2)
Takes as input:

* x1 -- input grid x1, shape (N1,)
* x2 -- input grid x2, shape (N2,)
* y -- input tensor, shape (...,N1,N2)
* xs1 -- interp grid, shape (Ns1,)
* xs2 -- interp grid, shape (Ns2,)

and outputs:

* ys -- interp tensor, shape (...,Ns1,Ns2)
