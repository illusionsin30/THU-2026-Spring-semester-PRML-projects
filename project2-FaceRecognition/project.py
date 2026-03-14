import os
import argparse
import logging

import numpy as np
import PIL
import sklearn
import matplotlib.pyplot as plt
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ================== utils ===================
def dataloader(dir_path):
    labels = os.listdir(dir_path)   # the path of data is like "face_data/0"
    train_data = []
    train_data_label = []
    test_data = []
    test_data_label = []
    for label in labels:
        sub_data = []
        sub_label = []
        img_dir_path = os.path.join(dir_path, label)
        imgs_paths = os.listdir(img_dir_path)
        
        for img_path in imgs_paths:
            img_path = os.path.join(img_dir_path, img_path)
            img = Image.open(img_path)  # img_size: 48 x 48 x 3 = 6912, so use pixels directly
            img = np.array(img).flatten() # [1, img_size]
            sub_data.append(img)
            sub_label.append(label)
        
        sub_data_arr = np.array(sub_data)
        sub_label_arr = np.array(sub_label)
        indices = np.arange(sub_data_arr.shape[0])
        
        # seperate train data and test data randomly
        np.random.shuffle(indices)
        sub_data_arr = sub_data_arr[indices]
        sub_label_arr = sub_label_arr[indices]
        # 250 for train, 50 for test
        train_data.append(sub_data_arr[:250])
        train_data_label.append(sub_label_arr[:250])
        test_data.append(sub_data_arr[-50:])
        test_data_label.append(sub_label_arr[-50:])
    
    train_data = np.concatenate(train_data)
    train_data_label = np.concatenate(train_data_label)
    test_data = np.concatenate(test_data)
    test_data_label = np.concatenate(test_data_label)

    return train_data, train_data_label, test_data, test_data_label


def load_svm(kernel, C, degree, tol, max_iter):
    svc_model = sklearn.svm.SVC(
        kernel=kernel,
        C=C,
        degree=degree,
        tol=tol,
        max_iter=max_iter
    )
    scaler = sklearn.preprocessing.StandardScaler()

    return svc_model, scaler


def plot_sv_figure(svs, sv_labels, save_path, gray, num_cols=4, num_figs=20):
    num_svs = svs.shape[0]
    num_svs = num_figs if num_figs < num_svs else num_svs
    num_rows = (num_svs + num_cols - 1) // num_cols
    plt.figure(figsize=(num_cols * 1.5, num_rows * 1.5))
    for i in range(num_svs):
        if i >= num_figs:
            break
        sv_img = svs[i]
        if gray:
            sv_img = np.dot(sv_img[...,:3], [0.299, 0.587, 0.114])
        
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(sv_img, cmap='gray')
        plt.title({f"Label: {sv_labels[i]}"})
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    logging.info(f"Figure has been saved at {save_path}.")


# ====== args processor ========
def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./face_data')
    parser.add_argument('--image_size', type=int, default=48)
    parser.add_argument('--C', type=float, default=1.0)
    parser.add_argument('--kernel', type=str, choices=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], default='linear')
    parser.add_argument('--degree', type=int, default=3)
    parser.add_argument('--tol', type=float, default=1e-3)
    parser.add_argument('--max_iteration', type=int, default=-1)
    parser.add_argument('--gray', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='./imgs')

    return parser


def main():
    args = create_argparser().parse_args()
    np.random.seed(args.seed)
    logging.info(f'Experiment settings: \
                 \n kernel: {args.kernel}, C: {args.C}, degree: {args.degree} \
                 \n tol: {args.tol}, max_iteration: {args.max_iteration}, seed: {args.seed}.')
    
    train_data, train_label, test_data, test_label = dataloader(args.data_dir)
    model, scaler = load_svm(args.kernel, args.C, args.degree, args.tol, args.max_iteration)
    
    scaled_train_data = scaler.fit_transform(train_data)
    scaled_test_data = scaler.fit_transform(test_data)
    logging.info(f"Training model...")
    model.fit(scaled_train_data, train_label)

    logging.info("Finished training. Start Evaluate.")
    acc = model.score(scaled_test_data, test_label)
    logging.info(f"The accuracy of model on test data: {acc:.4f}.")

    indices = model.support_
    np.random.shuffle(indices)
    svs = train_data[indices]
    sv_labels = train_label[indices]
    support_vectors = svs.reshape(-1, args.image_size, args.image_size, 3)
    logging.info(f"Plotting figures.")
    os.makedirs(args.save_path, exist_ok=True)
    save_path = f"{args.save_path}/sv_{args.kernel}_{args.C}.png"
    plot_sv_figure(support_vectors, sv_labels, save_path=save_path, gray=args.gray)


if __name__ == "__main__":
    main()