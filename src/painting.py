import numpy as np
import matplotlib.pyplot as plt


def Painting(list_picture, N=None, figsize=(15,6)):
    plt.figure(figsize=figsize)
    if N is None:
        N = len(list_picture[0])
    raws = len(list_picture)
    for k in range(N):
        for ind, cur_list in enumerate(list_picture):
            plt.subplot(raws, N, k+1+N*ind)
            plt.imshow(np.rollaxis(cur_list[k].numpy(), 0, 3), cmap='gray')
            plt.axis('off')
    plt.show()
