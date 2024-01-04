import numpy as np

RANGE = 16

'''
    `generate_invertible_matrix` generates a random integer matrix and ensures whether it 
    is invertible or not, by checking its rank. If the rank is not equal to the dimension n, 
    it generates a new matrix until an invertible one is found.
'''
def generate_invertible_matrix(n):
    # Generate a random n x n matrix with integer values
    matrix = np.random.randint(-RANGE, RANGE, size=(n,n))
    
    # Ensure that the matrix is invertible
    # Rank of a matrix is defined as the number of linearly independent rows in a matrix.
    # If a matrix is of the order 3×3 then the maximum rank of a matrix can be 3
    # A matrix A of dimension n x n is called invertible if and only if there exists another 
    # matrix B of the same dimension, such that AB = BA = I,
    while np.linalg.matrix_rank(matrix) != n:
        matrix = np.random.randint(-RANGE, RANGE, size=(n,n))
    
    return matrix

'''
    `reconstruct_matrix reconstructs` the original matrix from its eigenvalues and eigenvectors 
    using the formula A = P * Λ * P^-1, where A is the original matrix, Λ is a diagonal matrix 
    with eigenvalues, and P is the matrix with eigenvectors.
'''
def reconstruct_matrix(eigenvalues, eigenvectors):
    # Reconstruct the matrix from eigenvalues and eigenvectors
    inv_eigen_vector = np.linalg.inv(eigenvectors)
    Lambda = np.diag(eigenvalues)
    recon_matrix = np.dot(np.dot(eigenvectors, Lambda), inv_eigen_vector)
    return np.real(recon_matrix)

def check_reconstruction(matrix, recon_matrix):
    return np.allclose(matrix, recon_matrix)

def main():
    # Take input for matrix dimensions
    n = int(input("Enter the dimensions of the matrix(n): "))
    # Generate an invertible matrix
    matrix = generate_invertible_matrix(n)

    # Perform Eigen Decomposition
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Reconstruct matrix from eigenvalues and eigenvectors
    recon_matrix = reconstruct_matrix(eigenvalues, eigenvectors)

    print("\nOriginal Matrix:")
    print(matrix)
    print("\nReconstructed Matrix:")
    print(recon_matrix)

    # Check if the reconstruction worked properly
    if check_reconstruction(matrix, recon_matrix):
        print("\nReconstruction worked properly.")
    else:
        print("\nReconstruction didn't work properly!!")
        

if __name__ == "__main__":
    main()

