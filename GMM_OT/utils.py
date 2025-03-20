import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from scipy.linalg import sqrtm, inv
from torchvision import transforms
import torchvision
from IPython.display import display

# generate a GMM dataset
def generate_data(n_samples, n_classes, mu, sigma):
    np.random.seed(0)
    d = mu.shape[0]
    X = np.zeros((n_samples, d))
    for i in range(n_classes):
        mean = mu[i]
        cov = sigma[i] * np.eye(d)
        X[i * n_samples // n_classes:(i + 1) * n_samples // n_classes] = np.random.multivariate_normal(mean, cov, n_samples // n_classes)
    return X

# generate a half-moon datasets
def generate_half_moons(n_samples, noise=0.1):
    X, _ = make_moons(n_samples, noise=noise)
    return X

def plot_two_distrib(X0, X1):
    plt.figure(figsize=(12, 6))
    
    # Plot the first distribution
    plt.subplot(1, 2, 1)
    plt.scatter(X0[:, 0], X0[:, 1], color='blue', alpha=0.7, edgecolor='k')
    plt.title('First Distribution', fontsize=14)
    plt.xlabel('X1', fontsize=12)
    plt.ylabel('X2', fontsize=12)
    plt.grid(True)
    
    # Plot the second distribution
    plt.subplot(1, 2, 2)
    plt.scatter(X1[:, 0], X1[:, 1], color='blue', alpha=0.7, edgecolor='k')
    plt.title('Second Distribution', fontsize=14)
    plt.xlabel('X1', fontsize=12)
    plt.ylabel('X2', fontsize=12)
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Plot the original and transported distributions with arrows
def plot_transport(X, X_transported, title="Transported Points Visualization"):
    plt.figure(figsize=(8, 6))
    
    # Plot original points
    plt.scatter(X[:, 0], X[:, 1], color='blue', label="Original Points")
    
    # Plot transported points
    plt.scatter(X_transported[:, 0], X_transported[:, 1], color='red', label="Transported Points")
    
    # Draw arrows showing movement
    for i in range(len(X)):
        plt.arrow(X[i, 0], X[i, 1], 
                  X_transported[i, 0] - X[i, 0], 
                  X_transported[i, 1] - X[i, 1], 
                  color="gray", alpha=0.2, head_width=0.1, head_length=0.1)
    
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def sample_GMM(mu, Sigma, alpha, n_samples=1000):
    """Sample from a GMM model"""
    n_classes = len(alpha)
    d = mu.shape[1]
    X = np.zeros((n_samples, d))
    for i in range(n_classes):
        mean = mu[i]
        cov = Sigma[i]
        X[i * (n_samples // n_classes):(i + 1) * (n_samples // n_classes)] = np.random.multivariate_normal(mean, cov, n_samples // n_classes)
    return X

def plot_gmm_2d(mu_list, Sigma_list, alphas, xlim=(-5, 5), ylim=(-5, 5), resolution=100, ax=None, n_samples=None):
    """
    Plots the theoretical probability density function (PDF) of a 2D Gaussian Mixture Model (GMM) or a realization of the distribution.
    
    Parameters:
    - mu_list: List of mean vectors (each of shape (2,))
    - Sigma_list: List of covariance matrices (each of shape (2,2))
    - alphas: List of mixture weights (should sum to 1)
    - xlim, ylim: Limits of the plot
    - resolution: Number of grid points along each axis
    - ax: Matplotlib axis object (optional)
    - n_samples: Number of samples to draw from the GMM (if None, plot the theoretical PDF)
    """
    
    if n_samples is None:
        # Create grid for contour plot
        x = np.linspace(xlim[0], xlim[1], resolution)
        y = np.linspace(ylim[0], ylim[1], resolution)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))  # Shape (resolution, resolution, 2)
        
        # Compute GMM density
        gmm_pdf = np.zeros(X.shape)
        
        for mu, Sigma, alpha in zip(mu_list, Sigma_list, alphas):
            rv = multivariate_normal(mean=mu, cov=Sigma)
            gmm_pdf += alpha * rv.pdf(pos)  # Weighted sum of component densities

        if ax is None:
            # Plot contour levels
            plt.figure(figsize=(8, 6))
            plt.contourf(X, Y, gmm_pdf, levels=20, cmap='Blues')  # Filled contours
            plt.colorbar(label="Density")
            
            # Plot component means
            for mu in mu_list:
                plt.scatter(*mu, color='red', marker='o', edgecolors='black', s=1, label="Component Means")

            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.xlabel("X1")
            plt.ylabel("X2")
            plt.title("2D Gaussian Mixture Model (GMM)")
            plt.legend(["Component Means"])
            plt.grid()
            
            plt.show()
        
        else:
            ax.contourf(X, Y, gmm_pdf, levels=20, cmap='Blues')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel("X1")
            ax.set_ylabel("X2")
            ax.set_title("2D Gaussian Mixture Model (GMM)")
            ax.legend(["Component Means"])
            ax.grid()
    
    else:
        # Sample from the GMM
        X = sample_GMM(np.array(mu_list), np.array(Sigma_list), np.array(alphas), n_samples)
        
        if ax is None:
            plt.figure(figsize=(8, 6))
            plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
            plt.scatter(np.array(mu_list)[:, 0], np.array(mu_list)[:, 1], color='red', marker='o', edgecolors='black', s=10, label="Component Means")
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.xlabel("X1")
            plt.ylabel("X2")
            plt.title("Samples from 2D Gaussian Mixture Model (GMM)")
            plt.grid()
            plt.show()
        else:
            ax.scatter(X[:, 0], X[:, 1], alpha=0.5)
            ax.scatter(np.array(mu_list)[:, 0], np.array(mu_list)[:, 1], color='red', marker='o', edgecolors='black', s=10, label="Component Means")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel("X1")
            ax.set_ylabel("X2")
            ax.set_title("Samples from 2D Gaussian Mixture Model (GMM)")
            ax.grid()

# Displaying function for MNIST
def imshow(img,size=None):
    img = img*0.5 + 0.5     # unnormalize
    if size is not None:
      img = transforms.Resize(size=size, interpolation=transforms.InterpolationMode.NEAREST, antialias=True)(img)
    pil_img = torchvision.transforms.functional.to_pil_image(img)
    display(pil_img)
    # print("Image size (h x w): ",  pil_img.height, "x", pil_img.width)
    return None

def show_samples(y_1, y_2, y_3, y_4, num_samples=8, title="Generated AE Samples: Original 1 distribution (Top) vs. Transported 1 onto 6 (Bottom)"):
    fig, axes = plt.subplots(4, num_samples, figsize=(num_samples * 2, 6))

    for i in range(num_samples):
        # WGAN sample
        axes[0, i].imshow(y_1[i, 0].detach().cpu().numpy(), cmap='gray')
        axes[0, i].axis('off')

        # VAE sample
        axes[1, i].imshow(y_2[i, 0].detach().cpu().numpy(), cmap='gray')
        axes[1, i].axis('off')

        # AE sample
        axes[2, i].imshow(y_3[i, 0].detach().cpu().numpy(), cmap='gray')
        axes[2, i].axis('off')

        # Transported AE sample
        axes[3, i].imshow(y_4[i, 0].detach().cpu().numpy(), cmap='gray')
        axes[3, i].axis('off')

    plt.suptitle(title, fontsize=30)
    plt.show()
