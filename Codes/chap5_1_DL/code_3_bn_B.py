import numpy as np
from numpy import ndarray as array
from copy import deepcopy
from tqdm import tqdm


class MLP_BN:
    """该版本中，先做norm再做线性，符合BN原论文"""
    def __init__(self, L:int, ds:list, batch_size:int, momentum=0.99):
        assert len(ds) == L+1
        assert 0 < momentum and momentum < 1
        self.params = {"W":{}, "mu":{}, "sig":{}, "mu_all":{}, "sig_all":{}}
        for l in range(1, L+1):
            self.params["W"][l] = 0.1 * (np.random.rand(ds[l], ds[l-1]) - 0.5)
            self.params["mu"][l] = np.zeros([ds[l-1]])
            self.params["sig"][l] = np.zeros([ds[l-1]])
            self.params["mu_all"][l] = np.zeros([ds[l-1]])
            self.params["sig_all"][l] = np.zeros([ds[l-1]])
        self.ds, self.L, self.b_size = ds, L, batch_size
        self.ba_s, self.epsilon, self.momentum = [], 1e-6, momentum

    def forward(self, bx, is_train=True):
        assert bx.shape == (self.b_size, self.ds[0])
        self.ba_s, self.bz_s = [bx], [0] # self.bz_s[0] will never be visited
        mo = self.momentum
        for l in range(1, self.L+1):
            if is_train:
                mu_l = np.mean(self.ba_s[l-1], axis=0)
                sig_l = np.mean((self.ba_s[l-1] - mu_l)**2, axis=0)
                self.params["mu"][l], self.params["sig"][l] = mu_l, sig_l
                self.params["mu_all"][l] = mo * self.params["mu_all"][l] + (1-mo) * mu_l
                self.params["sig_all"][l] = mo * self.params["sig_all"][l] + (1-mo) * sig_l
            else:
                mu_l = self.params["mu_all"][l]
                sig_l = self.params["sig_all"][l]
            bz_l = (self.ba_s[l-1] - mu_l) / np.sqrt(sig_l + self.epsilon)
            W_l = self.params["W"][l]
            bzhat_l = np.einsum("ji,bi->bj", W_l, bz_l)
            self.bz_s.append(bz_l)
            self.ba_s.append(np.maximum(0, bzhat_l))
        return self.ba_s[self.L]
    
    def zero_grad(self):
        self.grads = {"W":{}}
        for l in range(1, self.L+1):
            self.grads["W"][l] = np.zeros_like(self.params["W"][l])

    def backward(self, bdLdy):
        assert bdLdy.shape == (self.b_size, self.ds[self.L],) # already mean at batch dimention
        bG_as = [bdLdy]
        for l in range(self.L, 0, -1):
            # calc bG_zh
            mu_l, sig_l, dl = self.params["mu"][l], self.params["sig"][l], self.ds[l]
            tmp = (self.ba_s[l] > 0).astype(np.int32) # (b, dl)
            bdadz = np.zeros([self.b_size, dl, dl]) #  dz / d batch \hat{z}
            for i in range(self.b_size):
                bdadz[i] = np.diag(tmp[i])
            bG_zh = np.einsum("bi,bij->bj", bG_as[-1], bdadz) # (b, dl)
            # calc bG_z
            bdzhdz = np.repeat(self.params["W"][l][None, :, :], self.b_size, axis=0)
            bG_z = np.einsum("bi,bij->bj", bG_zh, bdzhdz) # (b, d^{l-1})
            # calc bG_a
            scale = 1 / np.sqrt(sig_l + self.epsilon) # (dl, )
            bG_a_1 = bG_z * scale # (b, dl)
            bG_a_2 = - np.sum(bG_a_1 / self.b_size, axis=0)[None, :] # (1, dl)
            tmp = - np.sum((bG_z * (self.ba_s[l-1] - mu_l)), axis=0) * (scale**3) # (dl,)
            bG_a_3 = (self.ba_s[l-1] - mu_l) / self.b_size * tmp  # (b, dl)
            bG_a = bG_a_1 + bG_a_2 + bG_a_3
            bG_as.append(bG_a)
            # calc bG_W
            self.grads["W"][l] += bG_zh.T @ self.bz_s[l] # (dl, d^{l-1})
        pass



def BSGD_opt_MLP_BN(L_func, dL_func, mlp:MLP_BN, xs:array, ys:array, batch_size:int, 
                 loss_target:float, N_I:int=100, alpha_0=1e-4, gamma=0.9):
    def GD_update(param, grad, alpha):
        return param - alpha * grad
    N, N_b = xs.shape[0], xs.shape[0] // batch_size
    alpha, losses = alpha_0, []
    assert ys.shape[0] == N
    for k in range(N_I):
        ids = np.arange(N)
        np.random.shuffle(ids)
        for b in tqdm(range(N_b)):
            mlp.zero_grad()
            b_ids = ids[b*batch_size:(b+1)*batch_size]
            b_xs, b_ys = xs[b_ids], ys[b_ids]
            by_est = mlp.forward(b_xs, True)
            b_train_loss = L_func(b_ys, by_est)
            b_dLdy = dL_func(b_ys, by_est)
            mlp.backward(b_dLdy)
            b_train_loss = b_train_loss.mean()
            if b_train_loss < loss_target:
                mlp.zero_grad()
                print(f"Iter {k} batch loss = {b_train_loss:.4f} reach target")
                return
            for l in range(1, mlp.L+1):
                mlp.params["W"][l] = GD_update(
                    mlp.params["W"][l], mlp.grads["W"][l] / batch_size, alpha
                )
            pass
        alpha = alpha * gamma
        print(f"Iter {k}: loss = {b_train_loss}")
        losses.append(b_train_loss.copy())
    return losses    

