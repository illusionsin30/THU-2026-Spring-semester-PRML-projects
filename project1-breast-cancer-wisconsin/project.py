import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ======== calculate utils =========
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def between_class_scatter(w, mean_features_list):
    # for binary classification just use the direct index
    # mean_features.shape = (1, features_dim)
    diff = mean_features_list[1] - mean_features_list[0]
    Sb = diff.T @ diff
    b_scatter = w.T @ Sb @ w

    return Sb, b_scatter


def within_class_scatter(w, features_list, mean_features_list):
    # requires each mean_features in mean_features_list
    # is consistent with the features in features_list
    # features.shape = (num_samples, features_dim)
    # mean_features.shape = (1, features_dim)
    w_scatter = 0
    _, features_dim = features_list[0].shape
    Sw = np.zeros(shape=(features_dim, features_dim))
    for idx in range(len(mean_features_list)):
        features = features_list[idx]
        mean_features = mean_features_list[idx]
        diff = features - mean_features
        Sw_idx = diff.T @ diff
        Sw += Sw_idx
        w_scatter += w.T @ Sw_idx @ w
    
    return Sw, w_scatter


def L2_loss(pred_features, labels):
    loss = np.sum((pred_features - labels) ** 2) / 2

    return loss

# ======= data utils =======
def dataloader(file_path):
    # org data file doesn't need skiprows, but has ?
    # here we set '?' for 0
    # features.shape = (num_samples, features_dim=9)
    # label.shape = (num_samples, 1)
    data = np.genfromtxt(file_path, dtype=str)
    data[data == '?'] = '0'
    data_arr = data.astype(float)

    _, features_dim = data_arr.shape
    samples_code_number = data_arr[:, 0]
    features = data_arr[:, 1:(features_dim - 1)]
    label = data_arr[:, -1]

    return samples_code_number, features, label


# ========= model & train & eval ============
class BreastCancerPredictor:

    def __init__(self, features_dim, lr):
        self.features_dim = features_dim
        self.w = np.random.randn(self.features_dim, 1) * 0.01
        self.b = 0
        self.lr = lr
        self.loss_func = L2_loss


    def forward(self, x):
        # x.shape = (num_samples, features_dim)
        z = x @ self.w + self.b
        out = sigmoid(z)

        return out

    
    def backward(self, features, outs, label):
        partial_diff = (outs - label) * outs * (1 - outs)
        dw = features.T @ partial_diff
        db = np.sum(partial_diff)

        self.w -= self.lr * dw
        self.b -= self.lr * db


def train(model, features, label, epochs, tol):
    loss = float('inf')
    loss_list = []
    if epochs is None:
        epoch = 0
        while loss > tol:
            outs = model.forward(features)
            loss = L2_loss(outs, label)
            logging.info(f'Epoch: {epoch}, Now loss: {loss:.4f}')
            loss_list.append(loss)
            model.backward(features, outs, label)
            epoch += 1

    else:
        for epoch in range(epochs):
            outs = model.forward(features)
            loss = L2_loss(outs, label)
            logging.info(f'Epoch: {epoch}, Now loss: {loss:.4f}')
            loss_list.append(loss)
            model.backward(features, outs, label)

    return loss_list

def eval(model, features, label, threshold=0.5):
    pred_results = model.forward(features)
    pred_label = (pred_results >= threshold).astype(int)
    results = (pred_label == label)
    acc = np.mean(results)
    logging.info(f"Accuracy on test dataset: {acc}")

    return acc


# ========= plot utils ==========
def plot_loss_figure(loss_list, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_list, color='#1f77b4', label='L2 Loss', linewidth=2)
    
    plt.title('Loss', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    plt.close()


# ====== args processor ========
def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='./breast-cancer-wisconsin.txt')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--tol', type=float, default=8)
    parser.add_argument('--save_path', type=str, default='./loss.png')

    return parser

# ========== main ===========
def main():
    args = create_argparser().parse_args()
    logging.info(f"Hyperparameters setting: learning_rate {args.learning_rate} \
                 \nepochs {args.epochs} tol {args.tol}.")
    logging.info(f"Loading data from {args.file_path}")
    samples_code_number, features, label = dataloader(file_path=args.file_path)
    num_train_samples = int(args.train_ratio * features.shape[0])
    predict_model = BreastCancerPredictor(features_dim=9, lr=args.learning_rate)

    # extract features in different labels or class
    logging.info('Calculating for theoretic solution to Fisher criterion.')
    mask_benign = (label == 0)
    mask_malignant = (label == 1)
    features_benign = features[mask_benign]
    features_malignant = features[mask_malignant]
    features_list = [features_benign, features_malignant]
    
    # calculate the mean of features, attention the shape (1, features_dim)
    # that shape means features in code is a row vector rather than column vector
    mean_features_benign = np.mean(features_benign, axis=0).reshape(1, -1)
    mean_features_malignant = np.mean(features_malignant, axis=0).reshape(1, -1)
    mean_features_list = [mean_features_benign, mean_features_malignant]

    # calculate the theoretic solution to Fisher's criterion
    Sw, _ = within_class_scatter(predict_model.w, features_list, mean_features_list)
    diff_m = mean_features_malignant - mean_features_benign
    best_w = np.linalg.inv(Sw) @ diff_m.T
    best_w = best_w / np.linalg.norm(best_w)
    
    proj_benign = features_benign @ best_w
    proj_malignant = features_malignant @ best_w
    threshold = (proj_benign.mean() + proj_malignant.mean()) / 2
    proj_all = features @ best_w
    pred = (proj_all > threshold).astype(int)

    thm_acc = np.mean(pred == label)

    # fit the best_w by gradient descent, 
    # here choose L2 loss function
    logging.info(f"Training model...")
    label = label.reshape(-1, 1)
    train_features, train_label = features[:num_train_samples, :], label[:num_train_samples, :]
    test_features, test_label = features[num_train_samples:, :], label[num_train_samples:, :]
    loss_list = train(predict_model, train_features, train_label, args.epochs, args.tol)
    plot_loss_figure(loss_list, args.save_path)
    fit_best_w = predict_model.w

    logging.info(f"Evaluating model...")
    acc = eval(predict_model, test_features, test_label)
    
    cosine_similarity = np.dot(best_w.T, fit_best_w) / (np.linalg.norm(best_w) * np.linalg.norm(fit_best_w))
    logging.info(f"Theoretical best w*: {best_w}, Theoretical acc: {thm_acc}")
    logging.info(f"Fitted best w*: {predict_model.w}, Fitted best b*: {predict_model.b}")
    logging.info(f"Cosine similarity between theoretical w* and fitted w*: {cosine_similarity}.")

if __name__ == "__main__":
    main()