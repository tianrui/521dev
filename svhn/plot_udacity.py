import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.io import loadmat

def plot_udacity(stats_dir="./SVHN/udacity_stats.npz"):
    data = np.load(stats_dir)
    loss_vec = data['loss_vec']
    acc_vec = data['acc_vec']
    testloss_vec = data['testloss_vec']
    testacc_vec = data['testacc_vec']

    plt.plot(loss_vec, label='Training loss')
    plt.plot(acc_vec, label='Training accuracy')
    return

def plot_training_time():
    barwidth = 0.3
    ind = np.arange(2)
    t1 = [5224.89/60, 163.19]
    t2 = [14155.61/60, 269.51]
    fig, ax = plt.subplots()
    res1 = ax.bar(ind, [t1[0], t2[0]], barwidth)
    ax.set_ylabel('Time (min)')
    ax.set_title('CPU vs GPU on Convolutional Neural Net')
    ax.set_xticks(ind)
    ax.set_xticklabels(('GPU', 'CPU'))
    plt.savefig('barplot.png')
    plt.show()
    fig, ax = plt.subplots()
    res1 = ax.bar(ind, [t1[1], t2[1]], barwidth)
    ax.set_ylabel('Time (s)')
    ax.set_title('CPU vs GPU on fully-connected neural net')
    ax.set_xticks(ind)
    ax.set_xticklabels(('GPU', 'CPU'))
    plt.savefig('barplot_fc.png')
    plt.show()


if __name__ == "__main__":
    # plot_udacity()
    plot_training_time()