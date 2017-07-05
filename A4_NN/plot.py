import numpy as np
import matplotlib.pyplot as plt
import pdb

def plot_14():
    data = np.load('./14_vis.npz', 'r')
    train_vis = data['arr_0']
    # plot visualizations
    pdb.set_trace()
    plt.clf()
    plt.subplot(221)
    plt.imshow(train_vis[0], interpolation='nearest')
    plt.subplot(222)
    plt.imshow(train_vis[1], interpolation='nearest')
    plt.subplot(223)
    plt.imshow(train_vis[2], interpolation='nearest')
    plt.subplot(224)
    plt.imshow(train_vis[3], interpolation='nearest')
    plt.savefig('./part142_vis.jpg')
    plt.show()
    return

def plot_16():
    data = np.load('./14_vis.npz', 'r')
    train_vis = data['arr_0'].reshape((4, 5, 5, 32))
    j = 0
    # plot visualizations
    plt.clf()
    plt.subplot(221)
    plt.imshow(train_vis[0, :, :, j], interpolation='nearest')
    plt.subplot(222)
    plt.imshow(train_vis[1, :, :, j], interpolation='nearest')
    plt.subplot(223)
    plt.imshow(train_vis[2, :, :, j], interpolation='nearest')
    plt.subplot(224)
    plt.imshow(train_vis[3, :, :, j], interpolation='nearest')
    plt.savefig('./part142_vis.jpg')
    plt.show()
    return



if __name__ == '__main__':
    plot_16()
