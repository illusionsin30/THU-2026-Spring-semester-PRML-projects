import argparse
import functools

import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path='data.txt'):
    # X.shape = (num_samples, 2)
    # label.shape = (num_samples,)
    data = np.loadtxt(file_path)
    X = data[:, :-1]
    label = data[:, -1]

    return X, label

def find_weak_classifier(X, label, weights, x_range, y_range):
    assert X.shape[0] == weights.shape[0]
    direct = ['gt', 'lt']
    best_error = float('inf')
    best_classifier = None
    best_pred = None
    for x in x_range:
        for dir in direct:
            if dir == 'lt':
                pred = np.where(X[:, 0] < x, 1, -1)
            elif dir == 'gt':
                pred = np.where(X[:, 0] > x, 1, -1)
            error = np.sum(weights[pred != label])
            if error < best_error:
                best_error = error
                best_classifier = ('x', dir, x)
                best_pred = pred
    
    for y in y_range:
        for dir in direct:
            if dir == 'lt':
                pred = np.where(X[:, 1] < y, 1, -1)
            elif dir == 'gt':
                pred = np.where(X[:, 1] > y, 1, -1)
            error = np.sum(weights[pred != label])
            if error < best_error:
                best_error = error
                best_classifier = ('y', dir, y)
                best_pred = pred
    
    return best_classifier, best_error, best_pred

def predict_stump(X, classifier):
    dim_idx = 0 if classifier[0] == 'x' else 1
    dir = classifier[1]
    threshold = classifier[2]
    feature_col = X[:, dim_idx]
    if dir == 'gt':
        pred = np.where(feature_col > threshold, 1, -1)
    elif dir == 'lt':
        pred = np.where(feature_col < threshold, 1, -1)

    return pred

def final_classifier(classifiers, alphas, X):
    final_pred = np.zeros(X.shape[0])
    for classifier, alpha in zip(classifiers, alphas):
        pred = predict_stump(X, classifier)
        final_pred += alpha * pred

    return final_pred

class AdaBoost:
    def __init__(self, num_iterations):
        self.num_iterations = num_iterations
        self.classifiers = []
        self.alphas = []

        self.x_range = np.arange(1, 11)
        self.y_range = np.arange(2, 15)

    def fit(self, X, label):
        history_weights = []
        history_alphas = []
        self.weights = np.ones(X.shape[0]) / X.shape[0]
        history_weights.append(self.weights.copy())
        for i in range(self.num_iterations):
            print(f'Iteration {i + 1} / {self.num_iterations}.')
            self.iterate(X, label)
            history_weights.append(self.weights.copy())
        
        return functools.partial(final_classifier, self.classifiers, self.alphas), history_weights
        
    def iterate(self, X, label):
        classifier, error, pred = find_weak_classifier(X, label, self.weights, self.x_range, self.y_range)
        print(f'Best classifier: {classifier}, error: {error}.')
        self.classifiers.append(classifier)

        alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
        self.alphas.append(alpha)

        self.weights *= np.exp(-alpha * label * pred)
        self.weights /= np.sum(self.weights)

    def plot_boundary(self, X, label, final_classifier):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        plt.figure(figsize=(8, 6))
        Z = final_classifier(grid_points).reshape(xx.shape)
        plt.contour(xx, yy, Z, levels=[0], colors='k', linewidths=2, zorder=1)
        plt.scatter(X[:, 0], X[:, 1], c=label, edgecolors='k', zorder=3)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('AdaBoost Decision Boundary')
        plt.draw()
        plt.savefig('decision_boundary.png', dpi=300, bbox_inches='tight')
        plt.show()

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iterations', type=int, default=10)
    parser.add_argument('--data_path', type=str, default='data.txt')

    return parser

def plot_evo_figure(arr, xlabel, ylabel, title):
    plt.figure(figsize=(8, 6))
    if arr.ndim == 1:
        plt.bar(range(len(arr)), arr, label='Value')
    elif arr.ndim == 2:
        colors = plt.cm.viridis(np.linspace(0, 1, arr.shape[1]))
        for j in range(arr.shape[1]):
            plt.plot(arr[:, j], label=f'weight {j}', color=colors[j])
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.draw()
    plt.savefig(f'{title}.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    args = create_argparser().parse_args()
    X, label = load_data(args.data_path)
    model = AdaBoost(args.num_iterations)
    final_classifier, history_weights = model.fit(X, label)
    history_alphas, history_weights = np.array(model.alphas), np.array(history_weights)
    plot_evo_figure(history_weights, 'Iteration', 'Weight value', 'Weight_Evolution')
    plot_evo_figure(history_alphas, 'Iteration', 'Alpha value', 'Alpha_Evolution')

    print(f'alphas: {model.alphas}.')
    pred = np.sign(final_classifier(X))
    print(f'Predictions: {pred}.')
    acc = np.mean((pred == label).astype(int))
    print(f'Accuracy on data: {acc:.4f}')
    model.plot_boundary(X, label, final_classifier)
    
if __name__ == "__main__":
    main()