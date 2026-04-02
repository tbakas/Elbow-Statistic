from numpy import array, diff, sort, where, zeros, ones, argmax, arange
from numpy.random import uniform, seed
from numpy.linalg import svd
from matplotlib.pyplot import subplots, show
from sklearn.cluster import KMeans as skKMeans, AgglomerativeClustering as skAgglomerativeClustering
from sklearn.mixture import GaussianMixture as skGaussianMixture
from skfuzzy import cmeans


class ClusterModel:
    def __init__(self, K=range(1, 12), is_dummy=False):
        assert len(K) >= 3, 'Must have at least 3 cluster sizes'
        self.K = array(K)

        self.H = None
        self.H_type = ''
        self.delta = None
        self.p = None
        self.p_threshold_FWER = self.p_threshold_FDR = None
        self.significant_k_FWER = self.significant_k_FDR = None

        self.store_labels = not is_dummy
        self.Labels = None
        if self.store_labels:
            self.map = {k: i for i, k in enumerate(K)}
        else:
            self.map = None

    def fit(self, X):
        # This will determine cluster heterogeneity for each k in K
        # and also store the label results for non-dummy models.
        # This should also reset delta.
        self.Labels = None
        self.H = None
        self.delta = None

    def get_labels(self, k):
        return self.Labels[self.map[k]]

    def calculate_delta(self):
        # Delta is negative of second discrete derivative of H normalized by discrete derivative of H
        H_diff = diff(self.H)
        H_diff_diff = diff(H_diff)
        self.delta = - H_diff_diff / H_diff[1:]

    def _get_dummy(self):
        return self.__class__(self.K, is_dummy=True)

    def _get_thresholds(self, q1, q2):
        # calculates p-value thresholds for multiple hypothesis testing
        sorted_p = sort(self.p)
        m = self.p.shape[0]

        # Holm's Step-Down Procedure for controlling Family-Wise Error Rate
        j = 1
        while True:
            p = sorted_p[j - 1]
            if p > q1 / (m + 1 - j):
                p1 = p
                break
            j += 1
            if j == m + 1:
                # reject all H0
                p1 = 1.0
                break

        # Benjamini-Hochberg Procedure for controlling average False Discovery Rate
        j = m
        while True:
            p = sorted_p[j - 1]
            if p < q2 * j / m:
                p2 = p
                break
            j -= 1
            if j == 0:
                # reject no H0
                p2 = 0.0
                break

        return p1, p2

    def find_significant_k(self, X, q_FWER=0.05, q_FDR=0.05, N=100, use_pca=False, show_progress=True, set_rng=None):
        if self.delta is None:
            self.calculate_delta()

        if use_pca:
            _, _, V_transpose = svd(X)
            X = X @ V_transpose.T

        lower_bounds = X.min(axis=0)
        upper_bounds = X.max(axis=0)
        self.p = zeros(self.delta.shape)
        if set_rng is not None:
            seed(set_rng)
            
        dummy_model = self._get_dummy()
        
        # monte-carlo simulation to calculate p-values for delta statistics under
        # the null hypothesis the data (or its principle components) is uniformly distributed
        for i in range(N):
            if show_progress:
                print('Completed', i + 1, 'MC simulation(s) out of', N, end='\r')
            X_synthetic = uniform(low=lower_bounds, high=upper_bounds, size=X.shape)
            if use_pca:
                X_synthetic = X_synthetic @ V_transpose

            dummy_model.fit(X_synthetic)
            dummy_model.calculate_delta()
            self.p += where(dummy_model.delta >= self.delta, 1, 0)

        self.p = self.p / N
        self.p_threshold_FWER, self.p_threshold_FDR = self._get_thresholds(q_FWER, q_FDR)

        self.significant_k_FWER = []
        self.significant_k_FDR = []
        for p_value, k in zip(self.p, self.K[1:-1]):
            if p_value < self.p_threshold_FWER:
                self.significant_k_FWER.append(k)
            if p_value <= self.p_threshold_FDR:
                self.significant_k_FDR.append(k)

    def show_significant_k(self):
        print('FWER:', self.significant_k_FWER)
        print('FDR:', self.significant_k_FDR)

    def plot(self):
        if self.delta is None:
            self.calculate_delta()

        if self.p is None:
            figure, axes = subplots(2, sharex=True)
            axes[1].set_xlabel('$k$')
        else:
            figure, axes = subplots(3, sharex=True)
            axes[2].plot(self.K[1:-1], self.p, 'o-')
            axes[2].plot(self.K, self.p_threshold_FWER * ones(self.K.shape), '--', c='red')
            axes[2].plot(self.K, self.p_threshold_FDR * ones(self.K.shape), '--', c='green')
            axes[2].set_xlabel('$k$')
            axes[2].set_ylabel('p-value')
            axes[2].legend(['p-value', 'FWER', 'FDR'])

        axes[0].plot(self.K, self.H, 'o-')
        axes[0].set_xticks(self.K)
        axes[0].set_ylabel('$H$')
        axes[0].set_title(self.H_type)

        axes[1].plot(self.K[1:-1], self.delta, 'o-')
        axes[1].set_ylabel('$\delta$')

        figure.tight_layout()
        show()


class KMeans(ClusterModel):
    def __init__(self, K=range(1, 12), n_init='auto', is_dummy=False):
        super().__init__(K, is_dummy)
        self.n_init = n_init
        self.H_type = 'Inertia'

    def _get_dummy(self):
        return KMeans(self.K, self.n_init, True)

    def fit(self, X):
        self.delta = None
        if self.store_labels:
            self.Labels = zeros((self.K.shape[0], X.shape[0]), dtype=int)

        self.H = zeros(self.K.shape[0])
        for i, k in enumerate(self.K):
            clustering_model = skKMeans(k, n_init=self.n_init)
            clustering_model.fit(X)

            if self.store_labels:
                self.Labels[i] = clustering_model.labels_
            self.H[i] = clustering_model.inertia_


class Agglomerative(ClusterModel):
    def __init__(self, K=range(1, 12), linkage='ward', is_dummy=False):
        super().__init__(K, is_dummy)
        self.linkage = linkage
        self.H_type = 'Inertia'

    def _get_dummy(self):
        return Agglomerative(self.K, self.linkage, True)

    def __inertia(self, X, k, labels):
        groups = arange(k)
        centroids = zeros((k, X.shape[1]))
        for g in groups:
            in_group = where(labels == g, True, False)
            centroids[g] = X[in_group].mean(axis=0)

        return ((X - centroids[labels]) ** 2).sum()

    def fit(self, X):
        self.delta = None
        if self.store_labels:
            self.Labels = zeros((self.K.shape[0], X.shape[0]), dtype=int)
        self.H = zeros(self.K.shape[0])
        for i, k in enumerate(self.K):
            clustering_model = skAgglomerativeClustering(n_clusters=k, linkage=self.linkage)
            clustering_model.fit(X)

            if self.store_labels:
                self.Labels[i] = clustering_model.labels_
            self.H[i] = self.__inertia(X, k, clustering_model.labels_)


class GaussianMixture(ClusterModel):
    def __init__(self, K=range(1, 12), is_dummy=False):
        super().__init__(K, is_dummy)
        self.H_type = 'Negative Log-Likelihood'

    def fit(self, X):
        self.delta = None
        if self.store_labels:
            self.Labels = zeros((self.K.shape[0], X.shape[0]), dtype=int)
        self.H = zeros(self.K.shape[0])
        for i, k in enumerate(self.K):
            clustering_model = skGaussianMixture(n_components=k)
            clustering_model.fit(X)

            if self.store_labels:
                self.Labels[i] = clustering_model.predict(X)
            self.H[i] = -clustering_model.score(X)


class FuzzyCMeans(ClusterModel):
    def __init__(self, K=range(1, 12), m=2.0, error=0.001, maxiter=100, is_dummy=False):
        super().__init__(K, is_dummy)
        self.m = m
        self.error = error
        self.maxiter = maxiter
        self.H_type = 'Weighted Inertia'

    def _get_dummy(self):
        return FuzzyCMeans(self.K, self.m, self.error, self.maxiter, True)

    def fit(self, X):
        self.delta = None
        if self.store_labels:
            self.Labels = zeros((self.K.shape[0], X.shape[0]), dtype=int)
        self.H = zeros(self.K.shape[0])
        for i, k in enumerate(self.K):
            _, partitioned_matrix, _, _, objective, _, _ = cmeans(X.T, k, self.m, self.error, self.maxiter)

            if self.store_labels:
                self.Labels[i] = argmax(partitioned_matrix, axis=0)
            self.H[i] = objective[-1]
