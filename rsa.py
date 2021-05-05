import numpy as np

from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler

DISTINCT_COLORS = ["#e6194b", "#3cb44b", "#0082c8", "#f58230",
                   "#911eb4", "#46f0f0", "#f032e6", "#d2f53c", "#ffe119"]

def calculate_RDM(data, method = "correlation"):

    """ Return Specified distance matrices (1 = Pearson correlation,
    2 = Euclidian distance, 3 = Absolute activation difference)  """

    data = np.array(data)
    if method == "correlation":
        rdm = 1-np.corrcoef(data)
    elif method == "euclidean":
        # Use Eucledian distance
        rdm = pdist(data,'euclidean')
        rdm = squareform(rdm)
    elif method == "cosine":
        rdm = pdist(data,'cosine')
        rdm = squareform(rdm)
    elif method == "spearman":
        rdm = 1-spearmanr(data.transpose())[0]
    return rdm

def rdm_of_rdms(rdms):
    # 1. get only the upper triangle
    rdms = [rdm[np.triu_indices(len(rdm), 1)] for rdm in rdms]

    # 2. rdm of rdms
    rdm = calculate_RDM(rdms, "spearman")
    return rdm


def rsa(activations_per_model, method = "correlation"):

    # 1. get rdms of all different activation
    rdms = [calculate_RDM(activations, method) for activations in activations_per_model]

    # 2. rdm of rdms
    rdm = rdm_of_rdms(rdms)
    return rdm

def mds(matrix):
    scaler = MinMaxScaler()
    matrix_scaled = scaler.fit_transform(matrix)
    mds = MDS(2, random_state=0)
    return mds.fit_transform(matrix_scaled)


def plot_rsa(activations_per_model, labels = None, color = None, method = "correlation"):
    fig, axes = plt.subplots(1, 2, figsize = (10,5))

    r = rsa(activations_per_model, method)
    mat = axes[0].imshow(r)

    if labels != None:
        axes[0].set_xticks(range(len(labels)))
        axes[0].set_yticks(range(len(labels)))
        axes[0].set_xticklabels(labels, rotation='vertical')
        axes[0].set_yticklabels(labels)

    fig.colorbar(mat, ax=axes[0])

    m = mds(r)
    axes[1].scatter(m[:,0], m[:,1], label = labels, c = color)
    axes[1].set_xlabel("MDS 1")
    axes[1].set_ylabel("MDS 2")
    if labels != None:
        for i, txt in enumerate(labels):
            axes[1].annotate(txt, (m[i,0], m[i,1]))
    plt.show()

from mpl_toolkits.axes_grid1 import make_axes_locatable

class AxesDecorator():
    def __init__(self, ax, size="5%", pad=0.05, ticks=[1,2,3], ticks_label = [1,2,3], spacing=0.05,
                 color="k"):
        self.divider= make_axes_locatable(ax)
        self.ax = self.divider.new_vertical(size=size, pad=pad, sharex=ax, pack_start=True)
        ax.figure.add_axes(self.ax)
        self.ticks=np.array(ticks)
        self.d = np.mean(np.diff(ticks))
        self.spacing = spacing
        self.get_curve()
        self.color=color
        for x0 in ticks:
            self.plot_curve(x0)
        self.ax.set_yticks([])
        plt.setp(ax.get_xticklabels(), visible=False)
        self.ax.tick_params(axis='x', which=u'both',length=0)
        ax.tick_params(axis='x', which=u'both',length=0)
        for direction in ["left", "right", "bottom", "top"]:
            self.ax.spines[direction].set_visible(False)
        self.ax.set_xlabel(ax.get_xlabel())
        ax.set_xlabel("")
        self.ax.set_xticks(self.ticks)
        self.ax.set_xticklabels(ticks_label)
        print(ticks_label)

    def plot_curve(self, x0):
        x = np.linspace(x0-self.d/2.*(1-self.spacing),x0+self.d/2.*(1-self.spacing), 50 )
        self.ax.plot(x, self.curve, c=self.color)

    def get_curve(self):
        lx = np.linspace(-np.pi/2.+0.05, np.pi/2.-0.05, 25)
        tan = np.tan(lx)*10
        self.curve = np.hstack((tan[::-1],tan))
        return self.curve

def plot_rsa_fancy(activations_per_model, labels = None, color = None, method = "correlation", n_tasks = 1, title = "", steps=None):
    n_steps = len(steps)

    fig, axes = plt.subplots(1, 2, figsize = (8,4))

    r = rsa(activations_per_model, method)
    mat = axes[0].imshow(r)
    if title != "":
        plt.suptitle(title)

    axes[0].title.set_text("Representational dissimilarity matrix")
    axes[0].set_xticks(np.array(range(len(steps))) * 12 + 5.5)
    axes[0].set_yticks([])
    axes[0].set_xticklabels(steps, rotation='vertical')
    n_per_step = len(activations_per_model) / n_steps

    AxesDecorator(axes[0], ticks=np.array(range(len(steps))) * n_per_step + (n_per_step-1)/2.0, ticks_label = steps)
    #AxesDecorator(axes[0], ticks=np.array(range(3)) * n_per_step + (n_per_step-1)/2.0, ticks_label = steps)
    #axes[0].set_yticklabels()
    import matplotlib.ticker as tick
    fig.colorbar(mat, ax=axes[0], fraction=0.046, pad=0.04, format=tick.FormatStrFormatter('%.3f'))

    m = mds(r)
    axes[1].title.set_text("Multidimensional scaling")
    n_per_task_per_step = len(activations_per_model) / n_tasks / n_steps
    for j in range(n_tasks * len(steps)):
        base = int(j*n_per_task_per_step)
        fine_tunes = slice(int(j*n_per_task_per_step+1), int((j+1)*n_per_task_per_step))
        fine_tune_lines = slice(int(j*n_per_task_per_step), int((j+1)*n_per_task_per_step))

        axes[1].scatter(m[base][0], m[base][1], label = labels, c = "#000000", s = 40, marker = "*")

        if steps != None:
            axes[1].annotate(steps[int(j/n_tasks)], (m[base][0], m[base][1]), xytext=(m[base][0]*0.85+0.2, m[base][1]*0.85+0.2))

        axes[1].plot(m[fine_tune_lines][:,0], m[fine_tune_lines][:,1], c = "#000000", ls = '--', linewidth=0.25)

        axes[1].scatter(m[fine_tunes][:,0], m[fine_tunes][:,1], label = labels, c = DISTINCT_COLORS[j%n_tasks], s = 20, edgecolor='black', linewidth=0.25)

        max = np.max(np.absolute(m))
        max = max * 1.1

        axes[1].set_xlim(-max, max)
        axes[1].set_ylim(-max, max)
        axes[1].set_aspect('equal')
    #axes[1].scatter(m[0,0], m[0,1], label=labels, c="#000000", s=30)

    axes[1].set_xlabel("MDS 1")
    axes[1].set_ylabel("MDS 2")
    #for i, txt in enumerate(labels):
        #axes[1].annotate(txt, (m[i,0], m[i,1]))
    plt.show()

