#-----------Import packages--------------------

from typing import Tuple
import math
import numpy as np

import torch
from torch import Tensor


#-----------Identity transformation--------------------

class Identity():
    
    def __init__(self):
        self.name = 'ID'
    
    def __call__(self, data):
        return data
    
    def inv(self, data):
        return data


#-----------FFT transformation--------------------

class FFT():
    
    def __init__(self):
        self.name = 'FFT'
    
    def __call__(self, data):
        return torch.fft.fft(data)
    
    def inv(self, data):
        return torch.fft.ifft(data).real


#-----------DCT transformation--------------------

class DCT():
    
    def __init__(self):
        self.name = 'DCT'
    
    def __call__(self, data):
        return block_dct(data)
    
    def inv(self, data):
        return block_idct(data)


#-----------JPEG transformation--------------------

class JPEG():
    
    def __init__(self, block_size=8):
        self.name = 'JPEG'
        self.block_size = block_size
    
    def __call__(self, data):
        return batch_dct(data, self.block_size)
    
    def inv(self, data):
        return batch_idct(data, self.block_size)



#------LOG transformation ---------
class LOG():

    def __init__(self) :
        self.name = "LOG"
    
    def __call__(self,data):
        return log_transform(data)
    
    def inv(self, data):
        return inv_log(data)






#log transfromation functions  

def log_transform(data):
    c = 255 / np.log(2) 
    log_image = c*torch.log(1+ data)
    return log_image


#inverse log

def inv_log(data):
    c = 255 / np.log(2) 
    inv = torch.exp(data /c )-1
    return inv






#-----------torchjpeg.dct.__init__.py--------------------

def _normalize(N: int) -> Tensor:
    r"""
    Computes the constant scale factor which makes the DCT orthonormal
    """
    n = torch.ones((N, 1))
    n[0, 0] = 1 / math.sqrt(2)
    return n @ n.t()


def _harmonics(N: int) -> Tensor:
    r"""
    Computes the cosine harmonics for the DCT transform
    """
    spatial = torch.arange(float(N)).reshape((N, 1))
    spectral = torch.arange(float(N)).reshape((1, N))

    spatial = 2 * spatial + 1
    spectral = (spectral * math.pi) / (2 * N)

    return torch.cos(spatial @ spectral)


def block_dct(blocks: Tensor) -> Tensor:
    r"""
    Computes the DCT of image blocks
    Args:
        blocks (Tensor): Non-overlapping blocks to perform the DCT on in :math:`(N, C, L, H, W)` format.
    
    Returns:
        Tensor: The DCT coefficients of each block in the same shape as the input.
    Note:
        The function computes the forward DCT on each block given by 
        .. math::
            D_{i,j}={\frac {1}{\sqrt{2N}}}\alpha (i)\alpha (j)\sum _{x=0}^{N}\sum _{y=0}^{N}I_{x,y}\cos \left[{\frac {(2x+1)i\pi }{2N}}\right]\cos \left[{\frac {(2y+1)j\pi }{2N}}\right]
        
        Where :math:`i,j` are the spatial frequency indices, :math:`N` is the block size and :math:`I` is the image with pixel positions :math:`x, y`. 
        
        :math:`\alpha` is a scale factor which ensures the transform is orthonormal given by 
        .. math::
            \alpha(u) = \begin{cases}{
                    \frac{1}{\sqrt{2}}} &{\text{if }}u=0 \\
                    1 &{\text{otherwise}}
                \end{cases}
        
        There is technically no restriction on the range of pixel values but to match JPEG it is recommended to use the range [-128, 127].
    """
    N = blocks.shape[3]

    n = _normalize(N)
    h = _harmonics(N)

    if blocks.is_cuda:
        n = n.cuda()
        h = h.cuda()

    coeff = (2 / N) * n * (h.t() @ blocks @ h)

    return coeff


def block_idct(coeff: Tensor) -> Tensor:
    r"""
    Computes the inverse DCT of non-overlapping blocks
    Args:
        coeff (Tensor): The blockwise DCT coefficients in the format :math:`(N, C, L, H, W)`
    Returns:
        Tensor: The pixels for each block in the same format as the input.
    Note:
        This function computes the inverse DCT given by
        .. math::
            I_{x,y}={\frac {1}{\sqrt{2N}}}\sum _{i=0}^{N}\sum _{j=0}^{N}\alpha (i)\alpha (j)D_{i,j}\cos \left[{\frac {(2x+1)i\pi }{2N}}\right]\cos \left[{\frac {(2y+1)j\pi }{2N}}\right]
        See :py:func:`block_dct` for further details.
    """
    N = coeff.shape[3]

    n = _normalize(N)
    h = _harmonics(N)

    if coeff.is_cuda:
        n = n.cuda()
        h = h.cuda()

    im = (2 / N) * (h @ (n * coeff) @ h.t())
    return im


def batch_dct(batch: Tensor, block_size=8) -> Tensor:
    r"""
    Computes the DCT of a batch of images. See :py:func:`block_dct` for more details.
    This function takes care of splitting the images into blocks for the :py:func:`block_dct` and reconstructing
    the original shape of the input after the DCT.

    Args:
        batch (Tensor): A batch of images of format :math:`(N, C, H, W)`.

    Returns:
        Tensor: A batch of DCT coefficients of the same format as the input.

    Note:
        This fuction uses a block size of 8 to match the JPEG algorithm.
    """
    size = (batch.shape[2], batch.shape[3])

    im_blocks = blockify(batch, block_size)
    dct_blocks = block_dct(im_blocks)
    dct = deblockify(dct_blocks, size)

    return dct



def batch_idct(coeff: Tensor, block_size=8) -> Tensor:
    r"""
    Computes the inverse DCT of a batch of coefficients. See :py:func:`block_dct` for more details.
    This function takes care of splitting the images into blocks for the :py:func:`block_idct` and reconstructing
    the original shape of the input after the inverse DCT.

    Args:
        batch (Tensor): A batch of coefficients of format :math:`(N, C, H, W)`.

    Returns:
        Tensor: A batch of images of the same format as the input.

    Note:
        This function uses a block size of 8 to match the JPEG algorithm.
    """
    size = (coeff.shape[2], coeff.shape[3])

    dct_blocks = blockify(coeff, block_size)
    im_blocks = block_idct(dct_blocks)
    im = deblockify(im_blocks, size)

    return im


def fdct(im: Tensor) -> Tensor:
    r"""
    Convenience function for taking the DCT of a single image

    Args:
        im (Tensor): A single image of format :math:`(C, H, W)`

    Returns:
        Tensor: The DCT coefficients of the input in the same format.

    Note:
        This function simply expands the input in the batch dimension and then calls :py:func:`batch_dct` then removes
        the added batch dimension of the result.
    """
    return block_dct(im.unsqueeze(0)).squeeze(0)



def idct(coeff: Tensor) -> Tensor:
    r"""
    Convenience function for taking the inverse InversDCT of a single image

    Args:
        im (Tensor): DCT coefficients of format :math:`(C, H, W)`

    Returns:
        Tensor: The image pixels of the input in the same format.

    Note:
        This function simply expands the input in the batch dimension and then calls :py:func:`batch_idct` then removes
        the added batch dimension of the result.
    """
    return block_idct(coeff.unsqueeze(0)).squeeze(0)


def blockify(im: Tensor, size: int) -> Tensor:
    r"""
    Breaks an image into non-overlapping blocks of equal size.

    Parameters
    ----------
    im : Tensor
        The image to break into blocks, must be in :math:`(N, C, H, W)` format.
    size : Tuple[int, int]
        The size of the blocks in :math:`(H, W)` format.

    Returns
    -------
    A tensor containing the non-overlappng blocks in :math:`(N, C, L, H, W)` format where :math:`L` is the
    number of non-overlapping blocks in the image channel indexed by :math:`(N, C)` and :math:`(H, W)` matches
    the block size.

    Note
    ----
    If the image does not split evenly into blocks of the given size, the result will have some overlap. It
    is the callers responsibility to pad the input to a multiple of the block size, no error will be thrown
    in this case.
    """
    bs = im.shape[0]
    ch = im.shape[1]
    h = im.shape[2]
    w = im.shape[3]

    im = im.reshape(bs * ch, 1, h, w)
    im = torch.nn.functional.unfold(im, kernel_size=(size, size), stride=(size, size))
    im = im.transpose(1, 2)
    im = im.reshape(bs, ch, -1, size, size)

    return im



def deblockify(blocks: Tensor, size: Tuple[int, int]) -> Tensor:
    r"""
    Reconstructs an image given non-overlapping blocks of equal size.

    Args:
        blocks (Tensor): The non-overlapping blocks in :math:`(N, C, L, H, W)` format.
        size: (Tuple[int, int]): The dimensions of the original image (e.g. the desired output)
            in :math:`(H, W)` format.

    Returns:
        The image in :math:`(N, C, H, W)` format.

    Note:
        If the blocks have some overlap, or if the output size cannot be constructed from the given number of non-overlapping
        blocks, this function will raise an exception unlike :py:func:`blockify`.

    """
    bs = blocks.shape[0]
    ch = blocks.shape[1]
    block_size = blocks.shape[3]

    blocks = blocks.reshape(bs * ch, -1, int(block_size**2))
    blocks = blocks.transpose(1, 2)
    blocks = torch.nn.functional.fold(blocks, output_size=size, kernel_size=(block_size, block_size), stride=(block_size, block_size))
    blocks = blocks.reshape(bs, ch, size[0], size[1])

    return blocks
