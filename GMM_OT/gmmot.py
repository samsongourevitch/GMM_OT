import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from scipy.linalg import sqrtm, inv
import ot

def wasserstein_gaussian(mu0, Sigma0, mu1, Sigma1):
    """Compute squared Wasserstein-2 distance between two Gaussians."""
    mean_diff = np.linalg.norm(mu0 - mu1) ** 2
    sqrt_Sigma0 = sqrtm(Sigma0)
    sqrt_term = sqrtm(sqrt_Sigma0 @ Sigma1 @ sqrt_Sigma0)
    
    if np.iscomplexobj(sqrt_term):
        sqrt_term = sqrt_term.real  # Numerical correction
    
    trace_term = np.trace(Sigma0 + Sigma1 - 2 * sqrt_term)
    return mean_diff + trace_term

def compute_optimal_coupling(means0, covs0, weights0, means1, covs1, weights1):
    """Compute optimal transport coupling w* between two Gaussian Mixture Models."""
    K0, K1 = len(weights0), len(weights1)
    
    # Compute the cost matrix C (W2^2 distances)
    C = np.zeros((K0, K1))
    for k in range(K0):
        for l in range(K1):
            C[k, l] = wasserstein_gaussian(means0[k], covs0[k], means1[l], covs1[l])

    # Solve the optimal transport problem using EMD
    w_star = ot.emd(weights0, weights1, C)
    
    return w_star, C

def compute_A(Sigma0, Sigma1):
    """Compute the affine transport map parameters A_{k,l} for Gaussian transport."""
    sqrt_Sigma1 = sqrtm(Sigma1)
    Sigma_middle = sqrtm(sqrt_Sigma1 @ Sigma0 @ sqrt_Sigma1)

    if np.iscomplexobj(Sigma_middle):
        Sigma_middle = Sigma_middle.real  # Numerical correction

    sqrt_inv_Sigma_middle = sqrtm(inv(Sigma_middle))
    
    A_kl = sqrt_Sigma1 @ sqrt_inv_Sigma_middle @ sqrt_Sigma1
    return A_kl

def interpolate_gaussians(mu0, Sigma0, mu1, Sigma1, t):
    """Compute interpolated Gaussian parameters using displacement interpolation."""
    A_kl = compute_A(Sigma0, Sigma1)
    
    # Compute interpolated mean and covariance
    m_t = (1 - t) * mu0 + t * mu1
    I = np.eye(mu0.shape[0])
    K = ((1 - t) * I + t * A_kl)
    Sigma_t = K @ Sigma0 @ K
    
    return m_t, Sigma_t

def compute_interpolated_gmm(means0, covs0, weights0, means1, covs1, weights1, w_star, t):
    """Compute interpolated Gaussian Mixture Model at time t."""
    K0, K1 = len(weights0), len(weights1)
    
    # Compute interpolated mixture
    interpolated_means = []
    interpolated_covs = []
    interpolated_weights = w_star.flatten()  # Flatten for mixture weights

    for k in range(K0):
        for l in range(K1):
            m_t, S_t = interpolate_gaussians(means0[k], covs0[k], means1[l], covs1[l], t)
            interpolated_means.append(m_t)
            interpolated_covs.append(S_t)
    
    interpolated_means = np.array(interpolated_means)
    interpolated_covs = np.array(interpolated_covs)

    return interpolated_means, interpolated_covs, interpolated_weights

def compute_transport_gaussian(mu0, Sigma0, mu1, Sigma1, x):
    """Compute the transport map between two Gaussians."""
    inv_Sigma0 = inv(Sigma0)
    sqrtm_Sigma0_Sigma1 = sqrtm(Sigma0 @ Sigma1)
    return mu1[:, None] + inv_Sigma0 @ sqrtm_Sigma0_Sigma1 @ (x.T - mu0[:, None])

def transport_gmm_mean(means0, covs0, weights0, means1, covs1, weights1, w_star, X):
    """Transport a sample X from GMM 0 to GMM 1."""
    K0, K1 = len(weights0), len(weights1)

    num = 0
    denum = 0
    
    for k in range(K0):
        # get the density function of the k-th gaussian at x
        gaussian_density_k = np.exp(-0.5 * np.sum((X - means0[k]) @ inv(covs0[k]) * (X - means0[k]), axis=1))
        gaussian_density_k /= np.sqrt((2 * np.pi) ** 2 * np.linalg.det(covs0[k]))
        denum += weights0[k] * gaussian_density_k
        for l in range(K1):
            num += w_star[k, l] * gaussian_density_k * compute_transport_gaussian(means0[k], covs0[k], means1[l], covs1[l], X)

    return (num / denum).T

def transport_gmm_rand(means0, covs0, weights0, means1, covs1, weights1, w_star, X):
    """Transport a sample X from GMM 0 to GMM 1."""
    K0, K1 = len(weights0), len(weights1)
    N, D = X.shape  # N = number of points, D = data dimensionality

    # Compute Gaussian densities for all components of GMM0 at points X
    densities = np.zeros((K0, N))

    # Compute densities using scipy's optimized multivariate_normal
    densities = np.zeros((K0, N))
    for k in range(K0):
        mvn = multivariate_normal(mean=means0[k], cov=covs0[k])
        densities[k] = mvn.pdf(X)

    # Weighted sum for denominator (p(x))
    denum = np.dot(weights0, densities)  # (N,)

    # Compute joint probabilities p(k, l | x)
    probas = np.zeros((K0, K1, N))
    for k in range(K0):
        for l in range(K1):
            probas[k, l] = w_star[k, l] * densities[k] / denum
    
    probas = probas.reshape(K0 * K1, N)

    # Compute all possible transport maps
    transports = np.zeros((K0, K1, D, N))
    for k in range(K0):
        for l in range(K1):
            transports[k, l] = compute_transport_gaussian(means0[k], covs0[k], means1[l], covs1[l], X)
    
    transports = transports.reshape(K0 * K1, D, N)

    # Sample transported points for each X
    rng = np.random.default_rng()
    X_transported = np.array([
        rng.choice(transports[:, :, i], p=probas[:, i])
        for i in range(N)
    ])

    return X_transported

def transport_gmm_mode(means0, covs0, weights0, means1, covs1, weights1, w_star, X):
    """Transport a sample X from GMM 0 to GMM 1."""
    K0, K1 = len(weights0), len(weights1)
    N, D = X.shape  # N = number of points, D = data dimensionality

    # Compute Gaussian densities for all components of GMM0 at points X
    densities = np.zeros((K0, N))

    # Compute densities using scipy's optimized multivariate_normal
    densities = np.zeros((K0, N))
    for k in range(K0):
        mvn = multivariate_normal(mean=means0[k], cov=covs0[k])
        densities[k] = mvn.pdf(X)

    # Weighted sum for denominator (p(x))
    denum = np.dot(weights0, densities)  # (N,)

    # Compute joint probabilities p(k, l | x)
    probas = np.zeros((K0, K1, N))
    for k in range(K0):
        for l in range(K1):
            probas[k, l] = w_star[k, l] * densities[k] / denum
    
    probas = probas.reshape(K0 * K1, N)

    # Compute all possible transport maps
    transports = np.zeros((K0, K1, D, N))
    for k in range(K0):
        for l in range(K1):
            transports[k, l] = compute_transport_gaussian(means0[k], covs0[k], means1[l], covs1[l], X)
    
    transports = transports.reshape(K0 * K1, D, N)

    # Sample transported points for each X by choosing the max probability
    X_transported = np.array([
        transports[np.argmax(probas[:, i]), :, i]
        for i in range(N)
    ])

    return X_transported

def transport_gmm_rand_1(means0, covs0, weights0, means1, covs1, weights1, w_star, X):
    """Transport a sample X from GMM 0 to GMM 1."""
    K0, K1 = len(weights0), len(weights1)

    denum = 0

    probas = []
    for k in range(K0):
        # get the density function of the k-th gaussian at x
        gaussian_density_k = np.exp(-0.5 * np.sum((X - means0[k]) @ inv(covs0[k]) * (X - means0[k]), axis=1))
        gaussian_density_k /= np.sqrt((2 * np.pi) ** 2 * np.linalg.det(covs0[k]))
        denum += weights0[k] * gaussian_density_k
        for l in range(K1):
            probas.append(w_star[k, l] * gaussian_density_k)

    probas = np.array(probas)
    probas = probas/denum

    transports = []
    for k in range(K0):
        for l in range(K1):
            transports_kl = compute_transport_gaussian(means0[k], covs0[k], means1[l], covs1[l], X)
            transports.append(transports_kl)
    transports = np.array(transports)

    rng = np.random.default_rng()

    X_transported = []
    for i in range(len(X)):
        point_1 = rng.choice(transports[:, :, i], p=probas[:,i])
        X_transported.append(point_1)

    X_transported = np.array(X_transported)

    return X_transported

def transport_gmm_rand_2(means0, covs0, weights0, means1, covs1, weights1, w_star, X):
    """Transport a sample X from GMM 0 to GMM 1."""
    K0, K1 = len(weights0), len(weights1)
    N, D = X.shape  # N = number of points, D = data dimensionality

    # Compute log-densities for numerical stability
    log_densities = np.zeros((K0, N))
    for k in range(K0):
        mvn = multivariate_normal(mean=means0[k], cov=covs0[k])
        log_densities[k] = mvn.logpdf(X)

    # Weighted log-sum-exp for p(x) to avoid underflow
    log_weights = np.log(weights0)
    log_denum = np.logaddexp.reduce(log_weights[:, None] + log_densities, axis=0)

    # Compute log-probabilities for stability
    log_probas = np.zeros((K0, K1, N))
    for k in range(K0):
        for l in range(K1):
            log_probas[k, l] = np.log(w_star[k, l]) + log_densities[k] - log_denum

    # Convert log-probabilities back to normal scale
    probas = np.exp(log_probas)
    
    probas = probas.reshape(K0 * K1, N)

    # Compute all possible transport maps
    transports = np.zeros((K0, K1, D, N))
    for k in range(K0):
        for l in range(K1):
            transports[k, l] = compute_transport_gaussian(means0[k], covs0[k], means1[l], covs1[l], X)
    
    transports = transports.reshape(K0 * K1, D, N)

    # Sample transported points for each X
    rng = np.random.default_rng()
    X_transported = np.array([
        rng.choice(transports[:, :, i], p=probas[:, i])
        for i in range(N)
    ])

    return X_transported

def fit_and_transport(X_1, X_2, n_comp_1=10, n_comp_2=10, method='rand'):
    """Fit two GMMs to the data and transport the first GMM to the second one."""
    # fit a GMM to the data
    gmm_1 = GaussianMixture(n_components=n_comp_1)
    gmm_2 = GaussianMixture(n_components=n_comp_2)

    gmm_1.fit(X_1)
    # print('First distribution :\n mean =', gmm_1.means_, '\n covariance =', gmm_1.covariances_)

    gmm_2.fit(X_2)
    # print('Second distribution :\n mean =', gmm_2.means_, '\n covariance =', gmm_2.covariances_)
    means0 = gmm_1.means_
    covs0 = gmm_1.covariances_
    weights0 = gmm_1.weights_   

    means1 = gmm_2.means_
    covs1 = gmm_2.covariances_
    weights1 = gmm_2.weights_

    # Compute optimal transport plan w*
    w_star, C = compute_optimal_coupling(means0, covs0, weights0, means1, covs1, weights1)

     # Apply transport map to the first distribution
    if method == "rand":
        X_transported = transport_gmm_rand(means0, covs0, weights0, means1, covs1, weights1, w_star, X_1)
    elif method == "mean":
        X_transported = transport_gmm_mean(means0, covs0, weights0, means1, covs1, weights1, w_star, X_1)

    return X_transported