import time
import torch

conv = torch.nn.Conv1d(2, 4, 3, 1, 0, 1)

for p in conv.parameters():
    print(p.shape)

x = torch.tensor([
    [0.2, 0.4, -0.3, 1.0, 0.1, 0.3, 0.2],
    [0.3, 0.2,  0.1, -0.1, 0.1, 0.3, 0.2]
    ])

w = torch.tensor([
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.3, 0.0]
    ],
    [
        [0.0, 0.2, 0.1],
        [0.0, 0.1, 0.0]
    ],    
    [
        [1.0, 0.2, 0.1],
        [0.2, 0.1, 0.0]
    ],
    [
        [0.9, 0.2, 0.1],
        [0.0, 0.1, 0.1]
    ]
    ])

x = torch.rand(2,44100)
w = torch.rand(32,2,13)

print(x.shape, w.shape)

tic = time.perf_counter()
y = torch.nn.functional.conv1d(x.unsqueeze(0), w)
toc = time.perf_counter()

elapsed = toc - tic
print(y)
print(f"{elapsed:0.3f}sec")