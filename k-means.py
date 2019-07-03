from sklearn.datasets import load_sample_image
from matplotlib import pyplot as plt
import numpy as np

china = load_sample_image("china.jpg")

# height, width , RGB
print(china.shape)

# scale RGB 255-> 0 <-> 1
data = china / 255.0
data = data.reshape(427 * 640, 3)
print(data.shape)


def plot_pixels(data, title, colors=None, N=10000):

    if colors is None:
        colors = data

    # choose a random subset

    rng = np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    R, G, B = data[i].T

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # Figure 1: Green vs Red
    ax[0].scatter(R, G, color=colors, marker='.')
    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))

    # Figure 2: Blue vs Red
    ax[1].scatter(R, B, color=colors, marker='.')
    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))

    fig.suptitle(title, size=20);


# plotting dataset with large amount of data
plot_pixels(data, title='Input color space:16 million colors')
plt.savefig('plot_pixels')

# Fix numPy issues
import warnings;

warnings.simplefilter('ignore')


# cluster all colors to just 16 colors
from sklearn.cluster import MiniBatchKMeans

kmeans = MiniBatchKMeans(16)
kmeans.fit(data)
new_colors = kmeans.cluster_centers_[kmeans.predict(data)]

plot_pixels(data, colors=new_colors, title="Reduced colors: 16")
plt.savefig('new_colors')

china_recolored = new_colors.reshape(china.shape)

fig, ax = plt.subplots(1, 2, figsize=(16, 6), subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.5)
ax[0].imshow(china)
ax[0].set_title('Original Image', size=16)
ax[1].imshow(china_recolored)
ax[1].set_title('16-color Image', size=16)
plt.savefig('recolored_image')
