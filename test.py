# def getodd_lazy(lst):
#     for i in lst:
#         if i%2==1:
#             yield i
#
# data = range(10)
# l = getodd_lazy(data)
# for i in l:
#     print(i)

import torch
import numpy as np

a = np.array([[0.4,0.4,0.8],[0.4,0.2,0.1]])
b = torch.from_numpy(a)
b = torch.max(b, 1)
print(b)