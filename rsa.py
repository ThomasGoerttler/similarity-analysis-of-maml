import numpy as np

from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
from cka import linear_CKA

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


