import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.decomposition import FastICA, PCA, IncrementalPCA
from sklearn.random_projection import GaussianRandomProjection as RP
from sklearn.metrics import silhouette_score as sil_score
from scipy.linalg import pinv
from scipy import sparse
from sklearn.datasets import load_breast_cancer, load_digits
import warnings
warnings.filterwarnings("ignore")


def km(data, filename):
    sse = []
    scores = []
    for i in range(2, 18):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
        kmeans_labels = kmeans.predict(data)
        scores.append(sil_score(data, kmeans_labels))

    plt.figure(figsize=(10, 8))

    plt.subplot(211)
    plt.plot(range(2, 18), sse)
    plt.title("Sum of Squares vs. Clusters")
    plt.xlabel('Clusters')
    plt.ylabel('Sum Square Error')

    plt.subplot(212)
    plt.plot(range(2, 18), scores)
    plt.title("Silhouette vs. Clusters")
    plt.xlabel('Clusters')
    plt.ylabel('Silhouette')
    plt.tight_layout()

    plt.savefig(f"figures/{filename}")
    plt.close()


def em(data, filename):
    bic = []
    scores = []
    for i in range(2, 18):
        gmm = GaussianMixture(n_components=i)
        gmm.fit(data)
        bic.append(gmm.bic(data))
        gmm_labels = gmm.predict(data)
        scores.append(sil_score(data, gmm_labels))

    plt.figure(figsize=(10, 8))

    plt.subplot(211)
    plt.plot(range(2, 18), bic)
    plt.title("BIC vs. Clusters")
    plt.xlabel('Clusters')
    plt.ylabel('BIC')

    plt.subplot(212)
    plt.plot(range(2, 18), scores)
    plt.title("Silhouette vs. Clusters")
    plt.xlabel('Clusters')
    plt.ylabel('Silhouette')
    plt.tight_layout()

    plt.savefig(f"figures/{filename}")
    plt.close()


def reconstruction_error(algo, data):
    W = algo.components_
    if sparse.issparse(W):
        W = W.todense()
    p = pinv(W)
    reconstructed = ((p @ W) @ (data.T)).T
    errors = np.square(data - reconstructed)
    return np.nanmean(errors)


def pca_components(data, dataset, filename):
    pca = PCA(n_components=data.shape[1])
    pca.fit(data)
    var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3) * 100)

    plt.title('PCA Analysis for %s' % dataset)
    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('Number of Features')
    plt.plot(range(1, data.shape[1] + 1), var)
    plt.legend()

    plt.savefig(f"figures/{filename}")
    plt.close()

    feat = np.argmax(var > 95) + 1
    print(f"{dataset} Dataset Features: {data.shape[1]}    Reduced Features: {feat}")
    return feat


def ica_components(data, dataset, filename):
    dimensions = data.shape[1] + 1
    kurtosis = []
    for dim in range(2, dimensions):
        ica = FastICA(n_components=dim, max_iter=600)
        res = ica.fit_transform(data)
        tmp = pd.DataFrame(res)
        k = tmp.kurt(axis=0)
        kurtosis.append(k.abs().mean())

    feat = np.argmax(kurtosis)

    plt.title('ICA Analysis for %s' % dataset)
    plt.ylabel('Kurtosis')
    plt.xlabel('Number of Features')
    plt.plot(range(2, dimensions), kurtosis)

    plt.savefig(f"figures/{filename}")
    plt.close()

    print(f"{dataset} Dataset Features: {data.shape[1]}    Reduced Features: {feat + 2}")
    return feat


def rp_components(data, dataset, filename):
    dimensions = data.shape[1] + 1
    kurtosis = []
    lowerbound = []
    upperbound = []
    for dim in range(2, dimensions):
        kurts = []
        for t in range(100):
            rp = RP(n_components=dim)
            res = rp.fit_transform(data)
            tmp = pd.DataFrame(res)
            k = tmp.kurt(axis=0)
            kurts.append(k.abs().mean())
        kurtosis.append(np.mean(kurts))
        lowerbound.append(np.mean(kurts) - np.std(kurts))
        upperbound.append(np.mean(kurts) + np.std(kurts))

    feat = np.argmax(kurtosis)
    plt.title('RP Analysis for %s' % (dataset))
    plt.ylabel('Kurtosis')
    plt.xlabel('Number of Features')
    plt.plot(range(2, dimensions), kurtosis)
    plt.fill_between(range(2, dimensions), lowerbound, upperbound, facecolor='gray', alpha=0.1)

    plt.savefig(f"figures/{filename}")
    plt.close()

    print(f"{dataset} Dataset Features: {data.shape[1]}    Reduced Features: {feat + 2}")
    return feat


def ipca_components(data, dataset, filename):
    ipca = IncrementalPCA(n_components=data.shape[1])
    ipca.fit(data)
    var = np.cumsum(np.round(ipca.explained_variance_ratio_, decimals=3) * 100)

    plt.title('IPCA Analysis for %s' % dataset)
    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('Number of Features')
    plt.plot(range(1, data.shape[1] + 1), var)
    plt.legend()

    plt.savefig(f"figures/{filename}")
    plt.close()

    feat = np.argmax(var > 95) + 1

    print(f"{dataset} Dataset Features: {data.shape[1]}    Reduced Features: {feat}")
    return feat


def compare_km(dataset, datasets, filename):
    all_sse = []
    all_scores = []
    all_methods = []
    for d in datasets:
        data = d[1]
        sse = []
        scores = []

        for i in range(2, 18):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(data)
            sse.append(kmeans.inertia_)
            kmeans_labels = kmeans.predict(data)
            scores.append(sil_score(data, kmeans_labels))

        all_sse.append(sse)
        all_scores.append(scores)
        all_methods.append(d[0])

    title = "K-means of " + dataset + " Using Dimensionality Reduction"
    plt.figure(figsize=(10, 8))
    plt.suptitle(title, y=1.05, fontsize=16)
    plt.subplot(211)
    for i, sse in enumerate(all_sse):
        plt.plot(range(2, 18), sse, label=all_methods[i])
    plt.title("Sum of Squares vs. Clusters")
    plt.xlabel('Clusters')
    plt.ylabel('Sum Square Error')
    plt.legend()

    plt.subplot(212)
    for i, score in enumerate(all_scores):
        plt.plot(range(2, 18), score, label=all_methods[i])
    plt.title("Silhouette vs. Clusters")
    plt.xlabel('Clusters')
    plt.ylabel('Silhouette')
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"figures/{filename}")
    plt.close()


def compare_em(dataset, datasets, filename):
    all_bic = []
    all_scores = []
    all_dims = []
    all_methods = []
    for d in datasets:
        data = d[1]
        dimensions = data.shape[1] + 1
        all_dims.append(list(range(2, dimensions)))
        bic = []
        scores = []
        for i in range(2, dimensions):
            gmm = GaussianMixture(n_components=i)
            gmm.fit(data)
            bic.append(gmm.bic(data))
            gmm_labels = gmm.predict(data)
            scores.append(sil_score(data, gmm_labels))

        all_bic.append(bic)
        all_scores.append(scores)
        all_methods.append(d[0])

    title = "EM of " + dataset + " Using Dimensionality Reduction"
    fig = plt.figure(figsize=(10, 8))
    plt.suptitle(title, y=1, fontsize=16)

    plt.subplot(211)
    for i, bic in enumerate(all_bic):
        plt.plot(all_dims[i], bic, label=all_methods[i])
    plt.title("BIC vs. Components")
    plt.xlabel('Components')
    plt.ylabel('BIC')
    plt.legend()
    plt.subplot(212)
    for i, score in enumerate(all_scores):
        plt.plot(all_dims[i], score, label=all_methods[i])
    plt.title("Silhouette vs. Components")
    plt.xlabel('Components')
    plt.ylabel('Silhouette')
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"figures/{filename}")
    plt.close()


def plot_learning_curve(solver, X, y, title, filename):
    train_szs, train_scores, test_scores = learning_curve(solver, X, y, train_sizes=np.linspace(0.2, 1.0, 5))

    tr_means = np.mean(train_scores, axis=1)
    tr_stdev = np.std(train_scores, axis=1)
    tt_means = np.mean(test_scores, axis=1)
    tt_stdev = np.std(test_scores, axis=1)
    norm_train_szs = (train_szs / max(train_szs)) * 100.0

    plt.figure()
    plt.title(title)
    plt.xlabel('% of Training Data')
    plt.ylabel('Mean Accuracy Score')


    plt.plot(norm_train_szs, tr_means, 'o-', color='darkorange', label='train')
    plt.plot(norm_train_szs, tt_means, 'o-', color='navy', label='test')
    plt.fill_between(norm_train_szs, tr_means-tr_stdev, tr_means+tr_stdev, color='darkorange', alpha=0.1)
    plt.fill_between(norm_train_szs, tt_means-tt_stdev, tt_means+tt_stdev, color='navy', alpha=0.1)
    plt.legend(loc='best')

    plt.savefig(f"figures/NN_LC_{filename}")
    plt.close()


def components_scatter(n, x, type, filename):
    plt.figure(figsize=(6, 4))
    plt.title(f'{type} Components')

    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                continue
            plt.scatter(x[:, i], x[:, j])

    plt.savefig(f"figures/{filename}")
    plt.close()



# ======================== LOAD DATA ==============================

print("")
print("Load Datasets")
print("")

# LOAD DIGITS Dataset
digits = load_digits()
digitsX = digits.data
digitsy = digits.target
digitsX_scaled = scale(digitsX)

# LOAD BREAST CANCER Dataset
bc = load_breast_cancer()
bcX = bc.data
bcy = bc.target
bcX_scaled = scale(bcX)

# ======================== Basic clustering ==============================

km(digitsX, 'KM_Digits_Quality.png')
em(digitsX, 'EM_Digits_Quality.png')
km(bcX, 'KM_BC_Quality.png')
em(bcX, 'EM_BC_Quality.png')

# ======================== Dimentionality Reduction ==============================

print("")
print("Dimensionality Reduction")
print("")

# Digits
print('Digits PCA')
digits_pca_components = pca_components(digitsX_scaled, 'Digits', "PCA_digits.png")
digitsPCA = PCA(n_components=digits_pca_components).fit(digitsX_scaled)
print('PCA Reconstruction Error: ', reconstruction_error(digitsPCA, digitsX_scaled))

print('Digits ICA')
digits_ica_components = ica_components(digitsX_scaled, 'Digits', "ICA_digits.png")
digitsICA = FastICA(n_components=digits_ica_components).fit(digitsX_scaled)
print('ICA Reconstruction Error: ', reconstruction_error(digitsICA, digitsX_scaled))

print('Digits RP')
digits_rp_components = rp_components(digitsX_scaled, 'Digits', "RP_digits_100.png")
if digits_rp_components <= 0:
    digits_rp_components = 1
digitsRP = RP(n_components=digits_rp_components).fit(digitsX_scaled)
print('ICA Reconstruction Error: ', reconstruction_error(digitsRP, digitsX_scaled))

print('Digits IPCA')
digits_ipca_components = ipca_components(digitsX_scaled, 'Digits', "IPCA_digits.png")
digitsIPCA = IncrementalPCA(n_components=digits_ipca_components).fit(digitsX_scaled)
print('IPCA Reconstruction Error: ', reconstruction_error(digitsIPCA, digitsX_scaled))



# Breast Cancer
print('Breast Cancer PCA')
bc_pca_components = pca_components(bcX_scaled, 'Breast Cancer', "PCA_bc.png")
bcPCA = PCA(n_components=bc_pca_components).fit(bcX_scaled)
print('PCA Reconstruction Error: ', reconstruction_error(bcPCA, bcX_scaled))

print('Breast Cancer ICA')
bc_ica_components = ica_components(bcX_scaled, 'Breast Cancer', "ICA_bc.png")
bcICA = FastICA(n_components=bc_ica_components).fit(bcX_scaled)
print('ICA Reconstruction Error: ', reconstruction_error(bcICA, bcX_scaled))

print('Breast Cancer RP')
bc_rp_components = rp_components(bcX_scaled, 'Breast Cancer', "RP_bc_100.png")
if bc_rp_components <= 0:
    bc_rp_components = 1
bcRP = RP(n_components=bc_rp_components).fit(bcX_scaled)
print('ICA Reconstruction Error: ', reconstruction_error(bcRP, bcX_scaled))

print('Breast Cancer IPCA')
bc_ipca_components = ipca_components(bcX_scaled, 'Breast Cancer', "IPCA_bc.png")
bcIPCA = IncrementalPCA(n_components=bc_ipca_components).fit(bcX_scaled)
print('IPCA Reconstruction Error: ', reconstruction_error(bcIPCA, bcX_scaled))


# reduce digits dataset
digitsPCA = PCA(n_components=40).fit_transform(digitsX)
digitsICA = FastICA(n_components=20).fit_transform(digitsX)
digitsRP  = RP(n_components=2).fit_transform(digitsX)
digitsIPCA = IncrementalPCA(n_components=40).fit_transform(digitsX)

# components_scatter(2, digitsPCA, "PCA", "PCA_digits_components.png")
# components_scatter(2, digitsICA, "ICA", "ICA_digits_components.png")
# components_scatter(2, digitsRP, "RP", "RP_digits_components.png")

digits_data = [['Original', digitsX], ['PCA', digitsPCA], ['ICA', digitsICA], ['RP', digitsRP], ['IPCA', digitsIPCA]]

# BC Comparison of Dimensionality
bcPCA = PCA(n_components=10).fit_transform(bcX)
bcICA = FastICA(n_components=29).fit_transform(bcX)
bcRP  = RP(n_components=2).fit_transform(bcX)
bcIPCA = IncrementalPCA(n_components=10).fit_transform(bcX)

# components_scatter(2, bcPCA, "PCA", "PCA_bc_components.png")
# components_scatter(2, bcICA, "ICA", "ICA_bc_components.png")
# components_scatter(2, bcRP, "RP", "RP_bc_components.png")

bc_data = [['Original', bcX], ['PCA', bcPCA], ['ICA', bcICA], ['RP', bcRP], ['IPCA', bcIPCA]]

compare_km('Digits', digits_data, "KM_digits_DR_comparison.png")
compare_em('Digits', digits_data, "EM_digits_DR_comparison.png")
compare_km('BC', bc_data, "KM_bc_DR_comparison.png")
compare_em('BC', bc_data, "EM_bc_DR_comparison.png")


# ============ NN Reduction ================
print("")
print("Neural Network trained on DR Dataset (BC)")
print("")

bc_data = [['Original', bcX_scaled], ['PCA', PCA(n_components=9).fit_transform(bcX_scaled)],['ICA', FastICA(n_components=8).fit_transform(bcX_scaled)],['RP', RP(n_components=8).fit_transform(bcX_scaled)],['IPCA', IncrementalPCA(n_components=9).fit_transform(bcX_scaled)]]

for data in bc_data:
    dataset = data[0]
    print(f"NN: DR Type {dataset}")

    X_train, X_test, y_train, y_test = train_test_split(scale(data[1]), bcy, test_size=0.2)
    nn = MLPClassifier((5, 2), max_iter=2000, activation='logistic', solver='lbfgs', alpha=1)
    plot_learning_curve(nn, X_train, y_train, f"{dataset} Learning Curve", f"{dataset}.png")

    t0 = time()
    nn.fit(X_train, y_train)
    nn_time = time() - t0
    nn_pred = nn.predict(X_test)

    print(f""" 
        classifier: {nn}
        dataset: {dataset}
        time: {nn_time}
    """)

# Generate Datasets Through KM Clustering
kmeans = KMeans(n_clusters=2)
kmeans.fit(bcX_scaled)
kmeans_labels = kmeans.predict(bcX_scaled)

X_train, X_test, y_train, y_test = train_test_split(np.c_[bcX_scaled, kmeans_labels], bcy, test_size=0.2)
nn = MLPClassifier((5, 2), max_iter=2000, activation='logistic', solver='lbfgs', alpha=1)
plot_learning_curve(nn, X_train, y_train, f"KM NN Learning Curve", f"KM.png")

t0 = time()
nn.fit(X_train, y_train)
nn_time = time() - t0
nn_pred = nn.predict(X_test)

print(f"NN KM Fitting Time: {nn_time}")

# Generate Datasets Through EM Clustering
em = GaussianMixture(n_components=2)
em.fit(bcX_scaled)
em_labels = em.predict(bcX_scaled)

X_train, X_test, y_train, y_test = train_test_split(np.c_[bcX_scaled, em_labels], bcy, test_size=0.2)
nn = MLPClassifier((5, 2), max_iter=2000, activation='logistic', solver='lbfgs', alpha=1)
plot_learning_curve(nn, X_train, y_train, f"EM NN Learning Curve", f"EM.png")

t0 = time()
nn.fit(X_train, y_train)
nn_time = time() - t0
nn_pred = nn.predict(X_test)

print(f"NN EM Fitting Time: {nn_time}")