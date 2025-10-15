import gzip
import numpy as np
import matplotlib.pyplot as plt
from code_1_MLP import *
np.random.seed(42)


######################## library func ########################

def softmax(x):
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=1)[:, None]

def cross_entropy_loss(y_true, y_pred, batch_size=1):
    assert np.all(y_pred > 0), "y_pred contains non-positive values"
    assert y_true.shape == (batch_size,)
    assert y_pred.shape[0] == batch_size
    tmp = -np.log(y_pred[np.arange(batch_size), y_true])
    loss = np.sum(tmp) / batch_size
    return loss

def cross_entropy_and_softmax_grad(y_true, y_pred, batch_size=1):
    assert y_true.shape == (batch_size,)
    assert y_pred.shape[0] == batch_size
    grad = y_pred.copy()
    grad[np.arange(batch_size), y_true] -= 1
    grad /= batch_size
    return grad

def one_hot(y, num_classes):
    return np.eye(num_classes)[y]


def load_mnist():
    def read_images(path):
        with gzip.open(path, 'rb') as f:
            magic, num, rows, cols = np.frombuffer(f.read(16), dtype='>i4', count=4)
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows*cols)
            return images / 255.0  # 归一化到[0,1]
    
    def read_labels(path):
        with gzip.open(path, 'rb') as f:
            magic, num = np.frombuffer(f.read(8), dtype='>i4', count=2)
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels
    X_train = read_images('mnist/train-images-idx3-ubyte.gz')
    y_train = read_labels('mnist/train-labels-idx1-ubyte.gz')
    X_test = read_images('mnist/t10k-images-idx3-ubyte.gz')
    y_test = read_labels('mnist/t10k-labels-idx1-ubyte.gz')
    
    return X_train, y_train, X_test, y_test


######################## test code ########################

TEST_ALGO = "BSGD" # "GD", "SGD", "BSGD"

# 测试GD_opt_MLP函数
def test_gd_opt_mlp():
    # data
    X_train, y_train, X_test, y_test = load_mnist()
    sample_size = 1000
    X_train = X_train[:sample_size]
    y_train = y_train[:sample_size]

    # network & loss func
    mlp = MLP(L=2, ds=[784, 64, 10])
    def L_func(y_true, y_pred):
        y_pred = softmax(y_pred.reshape(1, -1))
        return cross_entropy_loss(np.array([y_true]), y_pred)
    def dL_func(y_true, y_pred):
        y_pred = softmax(y_pred.reshape(1, -1))
        grad = cross_entropy_and_softmax_grad(np.array([y_true]), y_pred)
        return grad.reshape(-1)
    
    # training
    if TEST_ALGO == "GD":
        losses = GD_opt_MLP(
            L_func=L_func,
            dL_func=dL_func,
            mlp=mlp,
            xs=X_train,
            ys=y_train,
            loss_target=0.01,   # 目标损失值
            N_I=10,             # 迭代次数
            alpha=0.1           # 学习率
        )
    elif TEST_ALGO == "SGD":
        losses = SGD_opt_MLP(
            L_func=L_func,
            dL_func=dL_func,
            mlp=mlp,
            xs=X_train,
            ys=y_train,
            loss_target=0.01,   # 目标损失值
            N_I=10,             # 迭代次数
            alpha=0.01           # 学习率
        )
    elif TEST_ALGO == "BSGD":
        losses = BSGD_opt_MLP(
            L_func=L_func,
            dL_func=dL_func,
            mlp=mlp,
            xs=X_train,
            ys=y_train,
            loss_target=0.01,   # 目标损失值
            batch_size=32,      # 批大小
            N_I=20,             # 迭代次数
            alpha=0.005           # 学习率
        )
    
    # test
    correct = 0
    test_size = 100
    for i in range(test_size):
        x = X_test[i]
        y = y_test[i]
        output = mlp.forward(x)
        pred = np.argmax(output)
        if pred == y:
            correct += 1
    accuracy = correct / test_size
    print(f"Test Accuracy: {accuracy:.2%}")
    return losses


if __name__ == "__main__":
    losses = test_gd_opt_mlp()
    print("Training losses:", losses)