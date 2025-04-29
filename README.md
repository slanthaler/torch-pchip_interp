# torch-pchip_interp
Differentiable piecewise Hermite spline interpolation in pytorch

Simple standalone file that allows for interpolation of general pytorch tensors on irregularly spaced 1d and 2d grids. In 2d, the data must be on a (N1,N2) grid, but it is not necessary to have equidistant grids or to be using the same number of grid points in the first and second spatial dimension.

### cubic_interp1d(x,y,xs): 
Takes as input: 

* x (Tensor): input grid of shape (N,)
* y (Tensor): input tensor y of shape (...,N), additional batch-(or other-)dimensions allowed
* xs (Tensor): interp grid xs of shape (Ns,)

and outputs:

* ys (Tensor): interpolated tensor of shape (...,Ns)


### cubic_interp2d(x1,x2,y,xs1,xs2)
Takes as input:

* x1 (Tensor): input grid x1 of shape (N1,)
* x2 (Tensor): input grid x2 of shape (N2,)
* y (Tensor): input tensor of shape (...,N1,N2)
* xs1 (Tensor): interp grid of shape (Ns1,)
* xs2 (Tensor): interp grid of shape (Ns2,)

and outputs:

* ys (Tensor): interp tensor of shape (...,Ns1,Ns2)
