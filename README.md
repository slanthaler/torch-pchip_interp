# torch-pchip_interp
Differentiable piecewise Hermite spline interpolation in pytorch

Simple standalone file that allows for interpolation of general pytorch tensors on irragular 1d and 2d grids. In 2d, the data must be on a regular grid, but it is not necessary to have equidistant grids or to be using the same number of grid points in the first and second spatial dimension.

### cubic_interp1d(x,y,xs): 
Takes as input: 

* input grid x   -- shape (N,)
* input tensor y -- shape (...,N)
* interp grid xs -- shape (Ns,)

and outputs:

* interpolated tensor ys -- shape (...,Ns)


### cubic_interp2d(x1,x2,y,xs1,xs2)
Takes as input:

* input grid x1   -- shape (N1,)
* input grid x2   -- shape (N2,)
* input tensor y  -- shape (...,N1,N2)
* interp grid xs1 -- shape (Ns1,)
* interp grid xs2 -- shape (Ns2,)

and outputs:

* interp tensor ys -- shape (...,Ns1,Ns2)
