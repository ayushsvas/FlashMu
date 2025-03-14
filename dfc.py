import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import tensorly as tl
tl.set_backend('pytorch')
print(f"Backend for tensor manipulation:", tl.get_backend())
from tensorly import tenalg
tenalg.set_backend('einsum')
from tensorly import plugins
plugins.use_opt_einsum()
print(f"Ensuring opt_einsum...")



class CPDecomposedWeights(nn.Module):
    def __init__(self, shape, rank, implementation = 'factorized'):
        super(CPDecomposedWeights, self).__init__()
        if isinstance(rank, float):
            rank = tl.validate_cp_rank(shape, rank)
            
        self.diagonal_core = nn.Parameter(torch.empty((rank,2), dtype = torch.float)).contiguous()
        self.factors = nn.ParameterList([nn.Parameter(torch.empty((s,rank,2), dtype = torch.float)).contiguous() for s in shape])
        self.implementation = implementation

        self.reset_parameters()

    def reset_parameters(self,):
            """
            Initialize parameters using a standard initialization method.
            """
            nn.init.xavier_uniform_(self.diagonal_core)
            for factor in self.factors:
                nn.init.xavier_uniform_(factor)
    
    def forward(self):
        if self.implementation == 'factorized':
              return torch.view_as_complex(self.diagonal_core), [torch.view_as_complex(factor) for factor in self.factors]
        elif self.implementation == 'reconstructed':
            core = torch.view_as_complex(self.diagonal_core)
            factors = [torch.view_as_complex(factor) for factor in self.factors]
            return tl.einsum("e, ae, be, ce, de -> abcd", core, *factors)
         

class DecomposableParameters(nn.Module):
    def __init__(self, shape, decomposition, rank, forward_implementation='reconstructed'):
    
        super(DecomposableParameters, self).__init__()

        self.decomposition = decomposition
        
        if self.decomposition == 'tucker':
            # get the shape of the core tensor based on the rank  
            self.rank = tl.tucker_tensor.validate_tucker_rank(shape, rank, fixed_modes=None)

            # Store forward implementation type
            self.forward_implementation = forward_implementation
            
            # Create core tensor parameter
            self.core = nn.Parameter(torch.empty(self.rank + [2], dtype=torch.float)).contiguous()
            
            # Create factor matrix parameters
            self.factors = nn.ParameterList([
                nn.Parameter(torch.empty((s, r, 2), dtype=torch.float)).contiguous()
                for (s, r) in zip(shape, self.rank)
            ])
        
        else:
            self.rank = list(shape)
             # Create core tensor parameter
            self.core = nn.Parameter(torch.empty(self.rank + [2], dtype=torch.float)).contiguous()
           
            # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self,):
        """
        Initialize parameters using a standard initialization method.
        """
        nn.init.xavier_uniform_(self.core)
        if self.decomposition == 'tucker':
            for factor in self.factors:
                nn.init.xavier_uniform_(factor)

    def reconstruct(self,core, factors):
        for i, factor in enumerate(factors):            
            core = torch.tensordot(factor, core, dims = ([1],[i]))
        return core.permute(dims = (3,2,1,0))
        
    def forward(self):
        core = torch.view_as_complex(self.core)
        
        if self.decomposition == 'tucker':
            factors = [torch.view_as_complex(factor) for factor in self.factors]
            if self.forward_implementation == 'factorized':
                return core, factors
            else:
                core = self.reconstruct(core, factors)
        return core
         

################################################################
# 2D fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, bias, dilate, fourier_interp, decomposition, tensor_rank, implementation, separable, mem_checkpoint = False):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.separable = separable
        # if self.separable:
        #     self.in_channels = 1
        # else:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.dilate = dilate
        self.fourier_interp = fourier_interp
        
        self.slices1 = (slice(None), slice(None), slice(0, self.modes1//2, self.dilate), slice(0, self.modes2//2, self.dilate))
        self.slices2 = (slice(None), slice(None), slice(-self.modes1//2, None, self.dilate), slice(0, self.modes2//2, self.dilate))

        if self.fourier_interp is True and self.dilate == 2:
            self.interpolate_real = nn.Conv2d(in_channels=2*self.out_channels, out_channels = self.out_channels, kernel_size=3, padding = 1)
            self.interpolate_imag = nn.Conv2d(in_channels=2*self.out_channels, out_channels = self.out_channels, kernel_size=3, padding = 1)
        if self.fourier_interp is True and self.dilate == 4:
            self.interpolate_real = nn.Conv2d(in_channels=2*self.out_channels, out_channels = self.out_channels, kernel_size=5, padding = 2)
            self.interpolate_imag = nn.Conv2d(in_channels=2*self.out_channels, out_channels = self.out_channels, kernel_size=5, padding = 2)

        self.non_linearity = nn.GELU()

        self.scale = (1 / (in_channels * out_channels))
    
        self.separable = separable
        self.implementation = implementation

        self.decomposition = decomposition 
        self.rank = tensor_rank
        
        if self.separable and self.decomposition == 'tucker':
            self.weights1 =DecomposableParameters((self.in_channels, self.modes1//(2*self.dilate), self.modes2//(2*self.dilate)), self.decomposition,
                                                    tensor_rank, self.implementation) # directly creates complex tensor

            self.weights2 =DecomposableParameters((self.in_channels, self.modes1//(2*self.dilate), self.modes2//(2*self.dilate)), self.decomposition,
                                                            tensor_rank, self.implementation)
        elif self.decomposition == 'tucker':

            self.weights1 =DecomposableParameters((self.in_channels, self.out_channels, self.modes1//(2*self.dilate), self.modes2//(2*self.dilate)), self.decomposition,
                                                    tensor_rank, self.implementation) # directly creates complex tensor

            self.weights2 =DecomposableParameters((self.in_channels, self.out_channels, self.modes1//(2*self.dilate), self.modes2//(2*self.dilate)), self.decomposition,
                                                            tensor_rank, self.implementation)
        
        if self.separable and self.decomposition == 'dense':
            self.weights1 =DecomposableParameters((self.in_channels, self.modes1//(2*self.dilate), self.modes2//(2*self.dilate)), self.decomposition,
                                                    tensor_rank, self.implementation) # directly creates complex tensor

            self.weights2 =DecomposableParameters((self.in_channels, self.modes1//(2*self.dilate), self.modes2//(2*self.dilate)), self.decomposition,
                                                            tensor_rank, self.implementation)
        elif self.decomposition == 'dense':

            self.weights1 =DecomposableParameters((self.in_channels, self.out_channels, self.modes1//(2*self.dilate), self.modes2//(2*self.dilate)), self.decomposition,
                                                    tensor_rank, self.implementation) # directly creates complex tensor

            self.weights2 =DecomposableParameters((self.in_channels, self.out_channels, self.modes1//(2*self.dilate), self.modes2//(2*self.dilate)), self.decomposition,
                                                            tensor_rank, self.implementation)
            
        if self.separable and self.decomposition == 'cp':
            self.weights1 = CPDecomposedWeights((self.in_channels, self.modes1//(2*self.dilate), self.modes2//(2*self.dilate)), tensor_rank, self.implementation)
            self.weights2 = CPDecomposedWeights((self.in_channels, self.modes1//(2*self.dilate), self.modes2//(2*self.dilate)), tensor_rank, self.implementation)

        elif self.decomposition == 'cp':
            self.weights1 = CPDecomposedWeights((self.in_channels, self.out_channels, self.modes1//(2*self.dilate), self.modes2//(2*self.dilate)), tensor_rank, self.implementation)
            self.weights2 = CPDecomposedWeights((self.in_channels, self.out_channels, self.modes1//(2*self.dilate), self.modes2//(2*self.dilate)), tensor_rank, self.implementation)

        

        self.mem_checkpoint = mem_checkpoint
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

        if bias is not None:
            self.bias = nn.Parameter(torch.randn(1, self.out_channels, 1, 1))
        else:
            self.bias = None
    # Complex multiplication
    def compl_mul2d(self, input, weights, dummy_input):
        # for w in weights[1]:
        #     print(w.shape)
        if self.decomposition == 'tucker':
            if self.implementation == 'factorized':
                if self.separable:
                    return tl.einsum("BIXY, ixy, Ii, Xx, Yy -> BIXY", input, weights[0], weights[1][0], weights[1][1], weights[1][2])
                else:
                    return tl.einsum("BIXY, ioxy, Ii, Oo, Xx, Yy -> BOXY", input, weights[0], weights[1][0], weights[1][1], weights[1][2], weights[1][3])
            elif self.implementation == 'reconstructed':
                return tl.einsum("BIXY,IOXY->BOXY", input, weights) # this line doesn't work for separated weights yet.
        if self.decomposition == 'cp':
            if self.implementation == 'factorized':
                if self.separable:
                    return tl.einsum("BIXY, i, Ii, Xi, Yi -> BIXY", input, weights[0], weights[1][0], weights[1][1], weights[1][2])
                else:
                    return tl.einsum("BIXY, o, Io, Oo, Xo, Yo -> BOXY", input, weights[0], weights[1][0], weights[1][1], weights[1][2], weights[1][3])
            elif self.implementation == 'reconstructed':
                return tl.einsum("BIXY,IOXY->BOXY", input, weights) # this line doesn't work for separated weights yet.
        elif self.decomposition == 'dense':
            if self.separable:
                return input*weights   
            else:
                return tl.einsum("BIXY,IOXY->BOXY", input, weights) 


    def interpolate(self, x):
        return torch.view_as_complex(
                    torch.concatenate(
                    (self.non_linearity(self.interpolate_real(torch.view_as_real(x)[..., 0]).unsqueeze(-1)),
                    self.non_linearity(self.interpolate_imag(torch.view_as_real(x)[..., 1]).unsqueeze(-1))),
                    dim = -1)
                    )


    def forward(self, x,):
        batchsize, channels, height, width = x.shape
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x.float())
        # print(x_ft.shape)
        # Multiply relevant Fourier modes
        out_ft = torch.zeros((batchsize, self.out_channels, height, width // 2 + 1), dtype=torch.cfloat,
                             device=x.device)
        # with torch.cuda.amp.autocast(enabled = False):
        # print(torch.cuda.max_memory_allocated()/1024**3)
        if self.mem_checkpoint:
            out_ft[self.slices1] = \
                checkpoint(self.compl_mul2d, x_ft[self.slices1],
                                                                self.weights1(), self.dummy_tensor)
            out_ft[self.slices2] = \
                checkpoint(self.compl_mul2d, x_ft[self.slices2],
                                            self.weights2(), self.dummy_tensor)
            # if self.fourier_interp and self.dilate > 1:
            # # Interpolate for the uprocessed modes 
            #     with torch.autocast(device_type = 'cuda', enabled = False):                    
            #         out_ft = checkpoint(self.interpolate(out_ft + x_ft)) # earlier this was concatenation
           
        else:
            out_ft[self.slices1] = \
                self.compl_mul2d(x_ft[self.slices1], 
                                self.weights1(), self.dummy_tensor)
        
            out_ft[self.slices2] = \
                self.compl_mul2d(x_ft[self.slices2], 
                                    self.weights2(), self.dummy_tensor)
        
        if self.fourier_interp and self.dilate > 1:
        # Interpolate for the uprocessed modes 
            with torch.autocast(device_type = 'cuda', enabled = False):
                # out_ft = torch.view_as_complex(
                #     torch.concatenate(
                #     (self.non_linearity(self.interpolate_real(torch.concatenate((torch.view_as_real(out_ft)[..., 0], torch.view_as_real(x_ft)[..., 0]), dim = 1)).unsqueeze(-1)),
                #     self.non_linearity(self.interpolate_imag(torch.concatenate((torch.view_as_real(out_ft)[..., 1], torch.view_as_real(x_ft)[..., 1]), dim = 1)).unsqueeze(-1))),
                #     dim = -1)
                #     )
                
                out_ft = self.interpolate(out_ft + x_ft) # earlier this was concatenation
                
                # out_ft = torch.view_as_complex(
                #     torch.concatenate(
                #     (self.non_linearity(self.interpolate_real((torch.view_as_real(out_ft)[..., 0] + torch.view_as_real(x_ft)[..., 0])).unsqueeze(-1)),
                #     self.non_linearity(self.interpolate_imag((torch.view_as_real(out_ft)[..., 1] + torch.view_as_real(x_ft)[..., 1])).unsqueeze(-1))),
                #     dim = -1)
                #     )


         # Return to physical space
        x_ft = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), dim = (-2,-1))

        if self.bias is not None:
            x_ft = x_ft + self.bias

        return x_ft




class f_block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_modes: tuple, bias: bool, dilate: int, fourier_interpolate: bool,
                 decomposition: str, rank: float, implementation: str, separable_spectral_conv: bool, mem_checkpoint: bool, 
                 skip: str = 'linear', bn = False, fno_block_precision: str = 'full'):
        super(f_block, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.modes1, self.modes2 = n_modes
        self.bias = bias
        self.dilate = dilate
        self.fourier_interpolate = fourier_interpolate
        self.decomposition = decomposition
        self.rank = rank
        self.implementation = implementation
        self.separable = separable_spectral_conv
        self.mem_checkpoint = mem_checkpoint

        self.skip = skip
        self.bn = bn
        self.fno_block_precision = fno_block_precision
        
        self.spectral_conv_block = nn.ModuleList()
        self.conv_block = nn.ModuleList()

        self.batch_norm = nn.BatchNorm2d(self.out_channels)

        self.spectral_conv_block.append(
            SpectralConv2d(self.in_channels, self.out_channels, self.modes1, self.modes2, self.bias, self.dilate, self.fourier_interpolate,
                                    self.decomposition, self.rank, self.implementation, self.separable, self.mem_checkpoint)
                                        )


        if skip == 'linear':
            self.conv_block.append(nn.Conv2d(in_channels = self.in_channels, out_channels = self.out_channels, kernel_size = 1, bias = False))
        
        if skip == 'conv3':
            self.conv_block.append(nn.Conv2d(in_channels = self.in_channels, out_channels = self.out_channels, kernel_size = 3,
                                              padding = 1, bias = False))
        
        

    def forward(self, x):
        for spectral_conv, conv in zip(self.spectral_conv_block, self.conv_block):
            x = spectral_conv(x) + conv(x)
            # print(x.dtype)
            x = F.gelu(x)
            if self.bn:
                x = self.batch_norm(x)
        return x




class c_block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: bool, dilations = [1, 3, 9], batch_norm = False):
        super(c_block, self).__init__()

        self.layers = nn.ModuleList()
        self.identity_layers = nn.ModuleList()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if padding:
            self.padding = int((self.kernel_size - 1)/2) 
            # print(self.padding)
        self.dilations = dilations


        for dilation_factor in self.dilations:
            self.layers.append(
                            nn.Sequential(
                                nn.Conv2d(in_channels = self.in_channels, out_channels = self.out_channels, 
                                        kernel_size = self.kernel_size, stride = 1, padding = dilation_factor*self.padding, dilation = dilation_factor),
                                nn.GELU()   
                                        )
                                )
            if batch_norm:
                self.layers.append(
                    nn.BatchNorm2d(self.out_channels)
                    )

            self.identity_layers.append(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias = False)
                )

            self.in_channels = self.out_channels
        
        


    def forward(self, x):
        for layer, identity_layer in zip(self.layers, self.identity_layers):
            
            x = layer(x) + identity_layer(x)
            # print(x.dtype)
        return x


class dfc(nn.Module):
    def __init__(self, in_channels:int, width:int, n_modes: tuple, bias: bool = True, spectral_dilation_fac: int = 1, fourier_interpolate: bool = False, 
                decomposition: str = 'dense', rank: float = 1, implementation: str = 'factorized', separable_fourier_layers: list = [False]*4, mem_checkpoint: bool = False,
                skip: str = 'conv3', batch_norm: str = False, fno_block_precision:str = 'full', lifting_channels:int = 128, projection_channels:int = 128,
                kernel_size: int = 5, padding: bool = True, dilations: list = [1, 3, 9], num_layers = 4):

        super(dfc, self).__init__()

        # properties for fourer subblock
        self.in_channels = in_channels
        self.out_channels = width
        self.n_modes = n_modes
        self.bias = bias
        self.dilate = spectral_dilation_fac
        self.fourier_interpolate = fourier_interpolate
        self.decomposition = decomposition
        self.rank = rank
        self.implementation = implementation
        self.separable = separable_fourier_layers
        self.mem_checkpoint = mem_checkpoint
        self.skip = skip
        self.bn = batch_norm
        self.fno_block_precision = fno_block_precision
       

        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels

        
        # properties for dcnn subblock
        self.kernel_size = kernel_size
        self.padding = padding 
        self.dilations = dilations
        

        self.layers = nn.ModuleList()

        self.lift = nn.Sequential(
                                    nn.Conv2d(self.in_channels, self.lifting_channels, 1),
                                    nn.Conv2d(self.lifting_channels, self.out_channels, 1)
        )    
        
        self.project = nn.Sequential(
                                    nn.Conv2d(self.out_channels, self.lifting_channels, 1),
                                    nn.Conv2d(self.lifting_channels, 1, 1)
        )
        for i in range(num_layers):
        
            self.layers.append(
                f_block(in_channels = self.out_channels, out_channels = self.out_channels, n_modes =  self.n_modes, bias = self.bias,
                        dilate = self.dilate, fourier_interpolate=self.fourier_interpolate,
                        decomposition=self.decomposition, rank = self.rank, implementation=self.implementation,
                        separable_spectral_conv=self.separable[i], mem_checkpoint=self.mem_checkpoint, skip = self.skip, 
                        bn=self.bn, fno_block_precision = self.fno_block_precision, 
                    )
            )
            if self.dilations is not None and self.kernel_size is not None:
                self.layers.append(
                    c_block(in_channels = self.out_channels, out_channels = self.out_channels, kernel_size = self.kernel_size,
                                padding = self.padding, dilations = self.dilations)
                )

        self.layers.append(f_block(in_channels = self.out_channels, out_channels = self.out_channels, n_modes =  self.n_modes, bias = self.bias,
                        dilate = self.dilate, fourier_interpolate=self.fourier_interpolate,
                        decomposition=self.decomposition, rank = self.rank, implementation=self.implementation,
                        separable_spectral_conv=self.separable[-1], mem_checkpoint=self.mem_checkpoint, skip = self.skip, 
                        bn=self.bn, fno_block_precision = self.fno_block_precision,))


        self.process = nn.Sequential(*self.layers)


    def forward(self, x):
        x = self.lift(x)
        x = self.process(x)
        x = self.project(x)
        return x
