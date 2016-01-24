import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from matplotlib.backends.backend_pdf import PdfPages
import itertools

def hist_plot(ax, data, x, hue, palette="GnBu_d"):

    data_x = data[x]
    data_hue = data[hue]
    hue_names = sorted(data_hue.unique())

    # in this case, long and object dtypes are treated as category
    category_dtypes = list("lO")
    is_category = data_x.dtypes.char in category_dtypes

    # set bin
    if is_category:
        le = LabelEncoder().fit(data_x)
        data_x = le.transform(data_x)
        n_classes = len(le.classes_)
        bins = np.arange(n_classes + 1.0) - 0.5
    else:
        bins = 10

    # draw
    vals = []
    color = sns.color_palette(palette, n_colors=len(hue_names))
    for label in hue_names:
        try:
            mask = np.array(data_hue == label)
            vals.append(np.array(data_x[mask]))
        except KeyError:
            vals.append(np.array([]))

    ax.hist(vals, color=color, bins=bins, histtype="barstacked", label=hue_names)
    ax.legend()
    ax.set(xlabel=x)

    if is_category:
        ax.set_xticks(np.arange(n_classes))
        ax.set_xticklabels(le.classes_)
        ax.set_xlim(-0.5, n_classes - 0.5)

def scatter_plot(ax, data, x, y, hue, palette="GnBu_d", max_n=10000, jitter=True):

    data_x = data[x]
    data_y = data[y]
    data_hue = data[hue]

    # in this case, long and object dtypes are treated as category
    category_dtypes = list("lO")
    is_category_x = data_x.dtypes.char in category_dtypes
    is_category_y = data_y.dtypes.char in category_dtypes
    hue_names = sorted(data_hue.unique())

    if is_category_x:
        le = LabelEncoder().fit(data_x)
        data_x = le.transform(data_x)
        classes_x = le.classes_

    if is_category_y:
        le = LabelEncoder().fit(data_y)
        data_y = le.transform(data_y)
        classes_y = le.classes_

    # jitter
    if jitter:
        jitter_std_x, jitter_std_y = None, None
        if is_category_x:
            jitter_std_x = 0.1
        if is_category_y:
            jitter_std_y = 0.1
        data_x = rand_jitter(data_x, jitter_std_x)
        data_y = rand_jitter(data_y, jitter_std_y)

    # sampling
    np.random.seed(71)
    N = len(data)
    sample_idx = np.random.choice(N, np.min([max_n, N]), replace=False)

    data_x = np.array(data_x)[sample_idx]
    data_y = np.array(data_y)[sample_idx]
    data_hue = np.array(data_hue)[sample_idx]

    # draw
    color = sns.color_palette(palette, n_colors=len(hue_names))
    for i, label in enumerate(hue_names):
        try:
            mask = np.array(data_hue == label)
            ary_x = data_x[mask]
            ary_y = data_y[mask]
            # rasterized=True to prevent slow pdf rendering
            ax.scatter(ary_x, ary_y, s=20, c=color[i],  marker='.', edgecolors='none', rasterized=True, label=label)
        except KeyError:
            print ("key error {}".format(label))

    ax.legend()
    ax.set(xlabel=x, ylabel=y)

    if is_category_x:
        ax.set_xticks(np.arange(len(classes_x)))
        ax.set_xticklabels(classes_x)
        ax.set_xlim(-0.5, len(classes_x) - 0.5)
    if is_category_y:
        ax.set_yticks(np.arange(len(classes_y)))
        ax.set_yticklabels(classes_y)
        ax.set_ylim(-0.5, len(classes_y) - 0.5)

# add jitter
def rand_jitter(arr, stdev=None):
    if stdev is None:
        stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def create_pdf(fname, plot_type, data, items):
    plt.close("all")  # just in case

    with PdfPages(fname) as pdf:
        nrows, ncols = 2, 2
        naxes = nrows * ncols
        for i, item in enumerate(items):
            ii = i % naxes
            r = ii / nrows
            c = ii % nrows
            if ii == 0:
                f, axes = plt.subplots(nrows, ncols, figsize=(16, 12))

            if plot_type == "hist":
                hist_plot(axes[r][c], data, item, "label")
            elif plot_type == "scatter":
                item_x, item_y = item
                scatter_plot(axes[r][c], data, item_x, item_y, "label")
            else:
                raise Exception

            print ("drew {}".format(item))
            if (i % naxes == naxes - 1) or (i == len(items) - 1):
                f.tight_layout()
                pdf.savefig(f)
                plt.close("all")
                print ("saved fig")

if __name__ == '__main__':
    train = pd.read_csv("../input/train.csv")
    train["is_train"] = True

    data = train
    data = data.rename(columns={"Response": "label"})  # 1-8

    hist_items = ["Product_Info_1", "Product_Info_2",  "Product_Info_3",  "Product_Info_4",
                  "Product_Info_5", "Product_Info_6",  "Product_Info_7",  "Ins_Age", "Ht", "Wt", "BMI"]
    create_pdf("hist.pdf", "hist", data, hist_items)

    scatter_cols = ["Product_Info_2", "Ins_Age", "Ht", "Wt", "BMI"]
    scatter_items = list(itertools.permutations(scatter_cols, 2))
    create_pdf("scatter.pdf", "scatter", data, scatter_items)