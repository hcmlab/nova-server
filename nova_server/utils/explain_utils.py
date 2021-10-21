import numpy as np


def getTopXpredictions(prediction, topLabels):

    prediction_class = []

    for i in range(0, len(prediction[0])):
        prediction_class.append((i, prediction[0][i]))

    prediction_class.sort(key=lambda x: x[1], reverse=True)

    return prediction_class[:topLabels]


def preprocess(X, net):
    X = X.copy()
    X = net["preprocess_f"](X)
    return X


def postprocess(X, color_conversion, channels_first):
    X = X.copy()
    X = iutils.postprocess_images(
        X, color_coding=color_conversion, channels_first=channels_first
    )
    return X


def image(X):
    X = X.copy()
    return ivis.project(X, absmax=255.0, input_is_postive_only=True)


def bk_proj(X):
    X = ivis.clip_quantile(X, 1)
    return ivis.project(X)


def heatmap(X):
    X = ivis.gamma(X, minamp=0, gamma=0.95)
    return ivis.heatmap(X)


def heatmapgnuplot2(X):
    X = np.abs(X)
    return ivis.heatmap(X, cmap_type="gnuplot2", input_is_postive_only=True)


def heatmapCMRmap(X):
    X = np.abs(X)
    return ivis.heatmap(X, cmap_type="CMRmap", input_is_postive_only=True)


def heatmapnipy_spectral(X):
    X = np.abs(X)
    return ivis.heatmap(X, cmap_type="nipy_spectral", input_is_postive_only=True)


def heatmap_rainbow(X):
    X = np.abs(X)
    return ivis.heatmap(X, cmap_type="rainbow", input_is_postive_only=True)


def heatmap_inferno(X):
    X = np.abs(X)
    return ivis.heatmap(X, cmap_type="inferno", input_is_postive_only=True)


def heatmap_viridis(X):
    X = np.abs(X)
    return ivis.heatmap(X, cmap_type="viridis", input_is_postive_only=True)


def heatmap_gist_heat(X):
    X = np.abs(X)
    return ivis.heatmap(X, cmap_type="gist_heat", input_is_postive_only=True)


def graymap(X):
    return ivis.graymap(np.abs(X), input_is_postive_only=True)
