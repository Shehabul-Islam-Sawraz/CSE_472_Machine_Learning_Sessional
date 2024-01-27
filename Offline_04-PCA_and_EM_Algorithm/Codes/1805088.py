import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

INPUT_FILE_PATH = "../Datasets/6D_data_points.txt"

class PCA:
    def __init__(self, data_matrix):
        self.data_matrix = data_matrix
        self.mean = np.mean(data_matrix, axis=0)
        self.std = np.std(data_matrix, axis=0)
        self.centered_data = self.center_data()
        self.covariance_matrix = self.calculate_covariance_matrix()
        self.eigenvalues, self.eigenvectors = self.calculate_eigens()
        
    def center_data(self):
        # return (self.data_matrix - self.mean) / self.std
        return (self.data_matrix - self.mean)
    
    def calculate_covariance_matrix(self):
        cov_matrix = (self.centered_data.T @ self.centered_data) / len(self.data_matrix)
        return cov_matrix
    
    def calculate_eigens(self):
        return np.linalg.eigh(self.covariance_matrix)
    
    def project_data(self, num_components=2):
        if num_components > self.data_matrix.shape[1]:
            raise ValueError("Number of principal components cannot be greater than feature dimensions.")
        
        sorted_indices = np.argsort(self.eigenvalues)[::-1]
        selected_indices = sorted_indices[:num_components]
        principal_axes = self.eigenvectors[:, selected_indices]
        
        projected_data = np.dot(self.centered_data, principal_axes)
        return projected_data

def plot_data(data, title="Data Plot", save_path=None):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], marker='o', color='b', label='Data Points')
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
        
        
class GaussianMixtureModel:
    def __init__(self, data, num_components, num_trials=1):
        self.data = data
        self.num_components = num_components
        self.num_trials = num_trials
        self.best_likelihood = None
        self.best_parameters = None
    
    def covariance_matrix(self, data):
        n = data.shape[0]
        mean = np.mean(data, axis=0)
        S = (data - mean).T.dot((data - mean)) / (n - 1)
        return S

    def initialize_parameters(self):
        # Random initialization of means and covariances
        indices = np.random.choice(len(self.data), size=self.num_components, replace=False)
        means = self.data[indices]
        covariances = [np.cov(self.data.T) for _ in range(self.num_components)]
        weights = np.ones(self.num_components) / self.num_components

        self.parameters = {'means': means, 'covariances': covariances, 'weights': weights}

    def e_step(self):
        # E-step: Calculate responsibilities
        responsibilities = np.zeros((len(self.data), self.num_components))
        for i in range(self.num_components):
            responsibilities[:, i] = self.parameters['weights'][i] * self.multivariate_normal_pdf(self.data, self.parameters['means'][i], self.parameters['covariances'][i])
        
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)

        return responsibilities

    def m_step(self, responsibilities):
        # M-step: Update parameters
        N_k = responsibilities.sum(axis=0)

        # Update means
        self.parameters['means'] = np.dot(responsibilities.T, self.data) / N_k[:, np.newaxis]

        # Update covariances
        for i in range(self.num_components):
            diff = self.data - self.parameters['means'][i]
            self.parameters['covariances'][i] = np.dot(responsibilities[:, i] * diff.T, diff) / N_k[i]

        # Update weights
        self.parameters['weights'] = N_k / len(self.data)

    def log_likelihood(self):
        pdf_values = np.zeros(self.data.shape[0])
        for j in range(self.num_components):
            pdf_values += self.parameters['weights'][j] * self.multivariate_normal_pdf(self.data, self.parameters['means'][j], self.parameters['covariances'][j])

        log_likelihood = np.log(pdf_values).sum()
        return log_likelihood

    def em_algorithm(self, iterations=100, save=False):
        plt.ion()  # Turn on interactive mode for live updating plots
        fig, ax = plt.subplots()

        for trial in range(self.num_trials):
            self.initialize_parameters()

            prev_likelihood = float('-inf')

            for iteration in range(iterations):  # You can adjust the number of iterations
                responsibilities = self.e_step()
                self.m_step(responsibilities)

                current_likelihood = self.log_likelihood()

                # Check for convergence
                if np.abs(current_likelihood - prev_likelihood) < 1e-3:
                    break

                prev_likelihood = current_likelihood

                # Plot the current state of the GMM
                self.plot_gmm(ax, iteration)
                plt.pause(0.01)

            # Track the best parameters and likelihood across trials
            if self.best_likelihood is None or current_likelihood > self.best_likelihood:
                self.best_likelihood = current_likelihood
                self.best_parameters = self.parameters.copy()

        plt.ioff()
        if save:
            plt.savefig("gmm_for_k_prime.png")
        else:
            plt.show()
        

    def multivariate_normal_pdf(self, x, mean, cov):
        # print(x.shape)
        # print(mean.shape)
        # print(cov.shape)
        if x.shape[1] != mean.shape[0] or (x.shape[1], x.shape[1]) != cov.shape:
            raise ValueError("The dimensions of inputs don't match!!")
        if np.linalg.det(cov) == 0:
            raise ValueError("The covariance matrix can't be singular!!")
        
        # Multivariate normal probability density function
        diff = x - mean
        inv_cov = np.linalg.pinv(cov)
        exponent = np.exp(-0.5 * np.sum(np.dot(diff, inv_cov) * diff, axis=1))
        # print(exponent)
        det_cov = np.linalg.det(cov)
        normalization = ((2 * np.pi) ** (-len(mean) / 2)) * (det_cov ** (-0.5))
        return normalization * exponent
        
    def plot_gmm(self, ax, iteration):
        ax.clear()
        
        ax.scatter(self.data[:, 0], self.data[:, 1], alpha=0.5, label='Data Points', c=self.assign_clusters(self.data), cmap='viridis', s=5)

        x = np.linspace(self.data[:, 0].min()*1.3, self.data[:, 0].max()*1.3, 100)
        y = np.linspace(self.data[:, 1].min()*1.3, self.data[:, 1].max()*1.3, 100)
        X, Y = np.meshgrid(x, y)
        
        for i in range(self.num_components):
            mean = self.parameters['means'][i]
            covariance = self.parameters['covariances'][i]

            # Plot ellipse for the Gaussian distribution
            self.plot_contour(ax, X, Y, mean, covariance, label=f'Component {i + 1}')

        ax.set_title(f'Iteration {iteration + 1}')
        ax.legend()
        plt.pause(0.01)

    def plot_contour(self, ax, X, Y, mean, covariance, label=None):
        Z = np.column_stack((X.ravel(), Y.ravel()))
        Z = multivariate_normal(mean, covariance, allow_singular=True).pdf(Z).reshape(X.shape)
        levels = sorted(np.max(Z)*np.exp(-0.5*np.arange(0, 3, 0.5)**2))
        ax.contour(X, Y, Z, levels=levels, cmap='viridis')
        
        if label:
            ax.text(mean[0], mean[1], label, color='r', fontsize=8, ha='center', va='center')
    
    def assign_clusters(self, data):
        # Assuming self.parameters contains the current model parameters
        num_components = len(self.parameters['means'])
        
        # Calculate the likelihood for each data point under each component
        likelihoods = np.zeros((len(data), num_components))
        for i in range(num_components):
            mean = self.parameters['means'][i]
            covariance = self.parameters['covariances'][i]
            likelihoods[:, i] = self.multivariate_normal_pdf(data, mean, covariance)

        # Assign each data point to the cluster with the highest likelihood
        cluster_assignments = np.argmax(likelihoods, axis=1)

        return cluster_assignments
    
def plot_likelihood_vs_k(k_values, likelihood_values):
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, likelihood_values, marker='o')
    plt.title('Convergence Log-Likelihood vs. Number of Components (K)')
    plt.xlabel('Number of Components (K)')
    plt.ylabel('Convergence Log-Likelihood')
    # plt.show()
    plt.savefig("Convergence Log-Likelihood.png")
        
def get_dataset():
    dataset = []
    with open(INPUT_FILE_PATH, "r") as file:
        input = file.readline()
        while input:  # Continue reading until the end of the file
            # Split the line into a list of values
            values = [float(x) for x in input.split(',')]
            
            # Append the values to the data list
            dataset.append(values)
            
            # Read the next line
            input = file.readline()
    return len(dataset[0]), np.array(dataset)

# Example usage
if __name__ == "__main__":
    # Load your dataset or generate a sample dataset
    num_features, dataset = get_dataset()

    if num_features > 2:
        pca_model = PCA(dataset)
        projected_data = pca_model.project_data(num_components=2)
        plot_data(projected_data, title="PCA Projection", save_path="pca_projection.png")
    else:
        projected_data = dataset
        plot_data(dataset[:, :2], title="Original Data Plot", save_path="original_data_plot.png")
        
        
    # Set the range of K values
    k_range = range(3, 9)

    # Store the best log-likelihood for each K
    best_likelihoods = []

    # Run EM algorithm for each K
    for k in k_range:
        best_trial_likelihood = None

        # Run multiple trials and pick the best log-likelihood
        for _ in range(5):
            gmm = GaussianMixtureModel(projected_data, num_components=k)
            gmm.em_algorithm(iterations=100)

            current_likelihood = gmm.best_likelihood

            if best_trial_likelihood is None or current_likelihood > best_trial_likelihood:
                best_trial_likelihood = current_likelihood

        best_likelihoods.append(best_trial_likelihood)

    # Plot convergence log-likelihood vs. K
    plot_likelihood_vs_k(k_range, best_likelihoods)

    # Choose K' based on the maximum log-likelihood
    k_prime = k_range[np.argmax(best_likelihoods)]

    # Plot the estimated GMM for K'
    gmm_for_k_prime = GaussianMixtureModel(projected_data, num_components=k_prime)
    gmm_for_k_prime.em_algorithm(save=True)