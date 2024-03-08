import numpy as np
from sklearn.decomposition import PCA

def get_length(n):
    start = int(np.sqrt(n))
    while start > 1:
        if n % start == 0:
            break
        else:
            start -= 1
    return start, n // start

def cut_image(X, num_image=4, shape=(112,92)):
    height, width = shape
    num_width, num_height = get_length(num_image)
    part_width = width // num_width
    part_height = height // num_height
    
    X_cut = []
    
    for i in range(X.shape[0]):
        image = X[i].reshape(shape)
        for k in range(num_width):
            for j in range(num_height):
                left = j * part_width
                upper = k * part_height
                right = left + part_width
                lower = upper + part_height

                # Crop the image to get the part
                part = image[upper:lower, left:right]

                pixels = np.reshape(part, [1, part_width * part_height])
                pixels = np.asarray(pixels)
                
                if k == 0 and j == 0:
                    X_i = pixels
                else:
                    X_i = np.hstack([X_i, pixels])
                
        if len(X_cut) == 0:
            X_cut = X_i
        else:
            X_cut = np.vstack([X_cut, X_i])

    return X_cut

class ModularPCA:
    def __init__(self, n_components, num_image=4, shape_image=(112,92)):
        self.n_components = n_components
        self.shape = shape_image
        self.num_image = num_image
        
    def fit(self, X):
        self.X = cut_image(X, self.num_image, self.shape)
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(self.X)
            
    def transform(self, X):
        X_cut = cut_image(X, self.num_image, self.shape)
        return self.pca.transform(X_cut)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.pca.fit_transform(self.X)