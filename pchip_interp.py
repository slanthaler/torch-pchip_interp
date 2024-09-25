'''
Cubic hermite interpolation in 1d and 2d.

Extended from [https://stackoverflow.com/questions/61616810/how-to-do-cubic-spline-interpolation-and-integration-in-pytorch]
'''
import torch

def h_poly(t):
    tt = t[None, :]**torch.arange(4, device=t.device)[:, None]
    A = torch.tensor([
        [1, 0, -3, 2],
        [0, 1, -2, 1],
        [0, 0, 3, -2],
        [0, 0, -1, 1]
    ], dtype=t.dtype, device=t.device)
    return A @ tt

def cubic_interp1d(x, y, xs):
    '''
    Cubic Hermite spline interpolation in 1d.

    Input: x - shape (N,)
           y - shape (..., N)
           xs - shape (Ns,)
    Output: ys - shape (..., Ns)
    '''
    assert x.max()>=xs.max() and x.min()<=xs.min(), 'xs is out of bounds'
    m = (y[...,1:] - y[...,:-1]) / (x[1:] - x[:-1])
    m = torch.cat([
        m[...,[0]], (m[...,1:]+m[...,:-1])/2, m[...,[-1]]
        ], dim=-1)
    idxs = torch.searchsorted(x[1:], xs)
    dx = (x[idxs + 1] - x[idxs])
    hh = h_poly((xs - x[idxs]) / dx)
    ys = hh[[0]] * y[...,idxs]        \
        + hh[[1]] * m[...,idxs] * dx   \
        + hh[[2]] * y[...,idxs+1]      \
        + hh[[3]] * m[...,idxs+1] * dx
    return ys


def cubic_interp2d(x1,x2,y,xs1,xs2):
    '''
    Cubic Hermite spline interpolation in 2d.

    nput: x1 - shape (N1,)
          x2 - shape (N2,)
          y - shape (...,N1,N2)
          xs1 - shape (Ns1,)
          xs2 - shape (Ns2,)
    Output: ys - shape (...,Ns1,Ns2)
    '''
    ys = cubic_interp1d(x1, y.transpose(-2,-1), xs1).transpose(-2,-1) # interpolation along first dimension
    ys = cubic_interp1d(x2, ys, xs2) # interpolation along second dimension
    return ys



if __name__=='__main__':
    import matplotlib.pyplot as plt

    def distort_fn(x):
        '''
        Function to distort grid.
        '''
        return (torch.sqrt(1 + 15*x)-1)/3
    
    def test_interp1d():
        '''
        Implements a convergence test.
        '''
        y_fn = lambda x: torch.cos(25*x)*torch.exp(-5*x)
        
        Ns = 2048
        xs = distort_fn( torch.linspace(0,1,Ns) )
        ys = y_fn(xs)
        
        Nvals = 8*2**torch.arange(6)
        err = []
        for N in Nvals:
            x = distort_fn( torch.linspace(0,1,N) )
            y = y_fn(x)
            ys_interp = cubic_interp1d(x,y,xs)
            err.append(
                (ys-ys_interp).abs().mean()
            )
        
            # make one plot 
            if N==32:
                plt.figure()
                plt.plot(xs,ys_interp.detach().numpy().squeeze(), label='interpolated')
                plt.plot(xs,ys.detach().numpy().squeeze(), 'k--', label='original')
                plt.plot(x,y.detach().numpy().squeeze(), 'ko', label='data')
                plt.legend()
                plt.suptitle('Illustration of 1d interpolation')
                
        plt.figure()
        plt.plot(Nvals, err, 'o-', label='error')
        plt.plot(Nvals, 1/Nvals**3,'k--', label='$N^{-3}$')
        plt.grid()
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('N')
        plt.ylabel('Error')
        plt.title('Convergence Plot 1d-Interpolation')


    
    def test_interp2d():
        '''
        Implements a convergence test.
        '''
        y_fn = lambda X1,X2: torch.cos(5*X1)*torch.sin(15*X2)*torch.exp(-3*X2)
        
        # compute reference solution
        Ns = 512
        xs1 = distort_fn( torch.linspace(0,1,Ns) )
        xs2 = torch.linspace(0,1,Ns//2)
        Xs1, Xs2 = torch.meshgrid(xs1, xs2, indexing='ij')
        ys = y_fn(Xs1,Xs2)
            
        Nvals = 8*2**torch.arange(5)
        err = []
        for N in Nvals:
            x1 = distort_fn( torch.linspace(0,1,N) )
            x2 = torch.linspace(0,1,N//2)
            X1, X2 = torch.meshgrid(x1,x2, indexing='ij')
            y = y_fn(X1,X2)
            ys_interp = cubic_interp2d(x1,x2,y,xs1,xs2)
            err.append(
                (ys-ys_interp).abs().mean()
            )
            
            # make one plot 
            if N==64:
                plt.figure(figsize=(12,5))
                
                plt.subplot(1,2,1)
                plt.pcolor(Xs1,Xs2,ys_interp.detach().numpy())
                plt.colorbar()
                plt.title('Interpolated')
                
                plt.subplot(1,2,2)
                plt.pcolor(Xs1,Xs2,ys.detach().numpy())
                plt.colorbar()
                plt.title('Original')

                plt.suptitle('Illustration of 2d interpolation')
                    
        plt.figure()
        plt.plot(Nvals, err, 'o-', label='error')
        plt.plot(Nvals, 70/Nvals**3,'k--', label='$N^{-3}$')
        plt.grid()
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('N')
        plt.ylabel('Error')
        plt.title('Convergence Plot 2d-Interpolation')
            
    #
    test_interp1d()
    test_interp2d()
    plt.show()
