import matplotlib.pyplot as plt 
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

def plot_best_match(x_test, X_train, y_train, y_test=None, shape=(62, 47), path=None):
    '''
    Plot the best match of a test image in the training set
    Input:
        x_test: 1D array, test image
        X_train: 2D array, training set
        y_train: 1D array, labels of the training set
        y_test: label of the test image (optional)
        shape: tuple, shape of the images
        path: str, path to save the plot (optional)
    '''

    # Reduce dimension
    pca = PCA(n_components=150, svd_solver='randomized')
    X_train_reduced = pca.fit_transform(X_train)
    x_test_reduced = pca.transform(x_test.reshape(1, -1))

    # Calculate Euclidean distance to find the best match
    distances = np.linalg.norm(x_test_reduced - X_train_reduced, axis=1)

    # Find 3 best matches
    index_min_distances = np.argsort(distances)[:3]
    
    # Plot the best matches
    fig = plt.figure()
    gs = fig.add_gridspec(3,3,width_ratios=[10.5,3,3])
    ax0 = fig.add_subplot(gs[:,0])
    ax11 = fig.add_subplot(gs[0,1])
    ax21 = fig.add_subplot(gs[1,1])
    ax31 = fig.add_subplot(gs[2,1])
    ax12 = fig.add_subplot(gs[0,2])
    ax22 = fig.add_subplot(gs[1,2])
    ax32 = fig.add_subplot(gs[2,2])

    # Using Logistic Regression to predict the identity of the test image
    lr = LogisticRegression(multi_class='ovr', solver='liblinear')
    lr.fit(X_train_reduced, y_train)
    y_predict = lr.predict(x_test_reduced)
    
    ax0.imshow(x_test.reshape(shape), cmap='gray')
    color = 'black'
    true_label = ''
    if y_test != None:
        true_label = ' (True label: ' + str(y_test)+')'
        if y_test != y_predict:
            color = 'red'
    ax0.set_title('Predict: ' + str(y_predict[0]) + true_label, y=-0.1, size=12, color=color)
    ax0.axis('off')

    ax11.imshow(X_train[index_min_distances[0]].reshape(shape), cmap='gray')
    ax11.axis('off')

    ax21.imshow(X_train[index_min_distances[1]].reshape(shape), cmap='gray')
    ax21.axis('off')

    ax31.imshow(X_train[index_min_distances[2]].reshape(shape), cmap='gray')
    ax31.axis('off')
    
    text_content = ['#', 'Name/ID: ', 'Distance: ']
    ax_text = [ax12, ax22, ax32]
    for i, text in enumerate(text_content):
        for j in range(3):
            if i == 0:
                add_text = str(j+1)
            if i == 1:
                add_text = str(y_train[index_min_distances[j]])
            if i == 2:
                add_text = str(round(distances[index_min_distances[j]], 3))    
            ax_text[j].text(0, 0.8 - i * 0.2, text + add_text, ha='left', va='center', transform=ax_text[j].transAxes)
    ax12.axis('off')
    ax22.axis('off')
    ax32.axis('off')
    
    ax32.set_title('Best match', x=-0.1, y=-0.32, size=12)
    plt.show()

    if path:
        fig.savefig(path)