
def center_crop(x, shape):
    start = (x.shape[-1]-shape[-1])//2
    stop  = start + shape[-1]
    return x[...,start:stop]

def causal_crop(x, shape):
    stop = x.shape[-1] - 1
    start = stop - shape[-1]
    return x[...,start:stop]
