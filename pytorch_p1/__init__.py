import torch
a=torch.tensor([[ 0.0396, -0.1693,  0.3225],
        [-0.6208, -0.5151, -0.3119],
        [-0.5024, -0.3287,  1.0781],
        [ 0.4385,  0.1337,  0.0060]])
print(a)
b=torch.max(a,1)
print('max is:',b)