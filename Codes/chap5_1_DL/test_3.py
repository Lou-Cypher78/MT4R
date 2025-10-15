import gzip
import numpy as np
from code_3_bn_A import * # 先做线性变换再BN
# from code_3_bn_B import * # 先做BN再线性变换
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

# 测试BSGD_opt_MLP_BN函数
def test_bn_mlp():
    # data
    X_train, y_train, X_test, y_test = load_mnist()
    sample_size = 1000
    BATCH_SIZE = 32
    X_train = X_train[:sample_size]
    y_train = y_train[:sample_size]

    # network & loss func
    mlp = MLP_BN(L=2, ds=[784, 128, 10], batch_size=BATCH_SIZE)
    def L_func(y_true, y_pred):
        y_pred = softmax(y_pred.reshape(BATCH_SIZE, -1))
        return cross_entropy_loss(y_true, y_pred, BATCH_SIZE)
    def dL_func(y_true, y_pred):
        y_pred = softmax(y_pred.reshape(BATCH_SIZE, -1))
        grad = cross_entropy_and_softmax_grad(y_true, y_pred, BATCH_SIZE)
        return grad
    
    # training
    losses = BSGD_opt_MLP_BN(
        L_func=L_func,
        dL_func=dL_func,
        mlp=mlp,
        xs=X_train,
        ys=y_train,
        loss_target=0.01,   # 目标损失值
        batch_size=BATCH_SIZE, # 批大小
        N_I=50,             # 迭代次数
        alpha_0=0.1,       # 学习率初值, BN就是得大一点
        gamma=0.99,         # 学习率衰减率
    )

    # test
    correct = 0
    test_size = y_test.shape[0] // BATCH_SIZE * BATCH_SIZE
    num_batches = test_size // BATCH_SIZE
    for i in range(num_batches):
        batch_start = i * BATCH_SIZE
        batch_end = batch_start + BATCH_SIZE
        x_batch = X_test[batch_start:batch_end]
        y_batch = y_test[batch_start:batch_end]
        outputs = mlp.forward(x_batch, False)
        preds = np.argmax(outputs, axis=1)
        correct += np.sum(preds == y_batch)
    accuracy = correct / test_size
    print(f"Test Accuracy: {accuracy:.2%}")
    return losses


if __name__ == "__main__":
    losses = test_bn_mlp()
    print("Training losses:", losses)