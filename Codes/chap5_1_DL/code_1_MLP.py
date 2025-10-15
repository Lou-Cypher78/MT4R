import numpy as np
from numpy import ndarray as array
from tqdm import tqdm


class MLP:
    def __init__(self, L:int, ds:list):
        assert len(ds) == L+1
        self.params = {"W":{}, "b": {},}
        for l in range(1, L+1):
            self.params["W"][l] = np.random.randn(ds[l], ds[l-1]) * np.sqrt(2./ds[l-1])
            self.params["b"][l] = np.zeros([ds[l]])
        self.ds, self.L = ds, L
        self.a_s = []

    def forward(self, x):
        self.a_s, self.z_s = [x], [0] # self.zs[0] will never be visited
        for l in range(1, self.L+1):
            W_l, b_l = self.params["W"][l], self.params["b"][l]
            self.z_s.append(W_l @ self.a_s[l-1] + b_l)
            self.a_s.append(np.maximum(0, self.z_s[l]))
        return self.a_s[self.L]
    
    def zero_grad(self):
        self.grads = {"W":{}, "b":{},}
        for l in range(1, self.L+1):
            self.grads["b"][l] = np.zeros_like(self.params["b"][l])
            self.grads["W"][l] = np.zeros_like(self.params["W"][l])

    def backward(self, dLdy):
        assert dLdy.shape == (self.ds[self.L],)
        G_as = [dLdy]
        for l in range(self.L, 0, -1):
            dadz = np.diag((self.z_s[l] > 0).astype(np.int32))
            dzda_ = self.params["W"][l]
            G_a_last = G_as[-1]
            G_as.append(G_a_last @ dadz @ dzda_)
            dzdb = np.eye(self.ds[l])
            self.grads["b"][l] += G_a_last @ dadz @ dzdb
            for i in range(self.ds[l]):
                dzdwi = np.zeros_like(self.params["W"][l]) 
                dzdwi[i, :] = self.a_s[l-1]
                self.grads["W"][l][i, :] += G_a_last @ dadz @ dzdwi
        pass


def GD_opt_MLP(L_func, dL_func, mlp:MLP, xs:array, ys:array, loss_target:float, N_I:int=100, alpha=1e-4):
    def GD_update(param, grad, alpha):
        return param - alpha * grad
    N, losses = xs.shape[0], []
    assert ys.shape[0] == N
    for k in range(N_I):
        mlp.zero_grad()
        train_loss = 0
        for i in tqdm(range(N)):
            y_est = mlp.forward(xs[i])
            train_loss += L_func(ys[i], y_est)
            dLdy = dL_func(ys[i], y_est)
            mlp.backward(dLdy)
        train_loss = train_loss / N
        if train_loss < loss_target:
            print(f"Iter {k} loss = {train_loss:.4f} reach target")
            break
        for p, l in zip(mlp.params, range(1, mlp.L+1)):
            mlp.params[p][l] = GD_update(
                mlp.params[p][l], mlp.grads[p][l] / N, alpha
            )
        print(f"Iter {k}: loss = {train_loss}")
        losses.append(train_loss)
    return losses   


def SGD_opt_MLP(L_func, dL_func, mlp:MLP, xs:array, ys:array, loss_target, N_I:int=100, alpha=1e-4):
    def GD_update(param, grad, alpha):
        return param - alpha * grad
    N, losses = xs.shape[0], []
    assert ys.shape[0] == N
    for k in range(N_I):
        mlp.zero_grad()
        train_loss = 0
        for i in tqdm(range(N)):
            y_est = mlp.forward(xs[i])
            train_loss += L_func(ys[i], y_est)
            dLdy = dL_func(ys[i], y_est)
            mlp.backward(dLdy)
            for p, l in zip(mlp.params, range(1, mlp.L+1)):
                mlp.params[p][l] = GD_update(
                    mlp.params[p][l], mlp.grads[p][l], alpha
                )
            mlp.zero_grad()
        train_loss = train_loss / N
        print(f"Iter {k}: loss = {train_loss}")
        losses.append(train_loss)
        if train_loss < loss_target:
            print(f"Iter {k} loss = {train_loss:.4f} reach target")
            break
    return losses  



def BSGD_opt_MLP(L_func, dL_func, mlp:MLP, xs:array, ys:array, loss_target:float, batch_size:int, N_I:int=100, alpha=1e-4):
    def GD_update(param, grad, alpha):
        return param - alpha * grad
    N, N_b, losses = xs.shape[0], xs.shape[0] // batch_size, {}
    assert ys.shape[0] == N
    for k in range(N_I):
        ids = np.arange(N)
        np.random.shuffle(ids)
        losses[k] = []
        for b in tqdm(range(N_b)):
            mlp.zero_grad()
            b_ids = ids[b*batch_size:(b+1)*batch_size]
            b_train_loss, b_xs, b_ys  = 0, xs[b_ids], ys[b_ids]
            for i in range(batch_size):
                y_est = mlp.forward(b_xs[i])
                b_train_loss += L_func(b_ys[i], y_est)
                dLdy = dL_func(b_ys[i], y_est)
                mlp.backward(dLdy)
            b_train_loss = b_train_loss / batch_size
            if b_train_loss < loss_target:
                print(f"Iter {k} batch loss = {b_train_loss:.4f} reach target")
                mlp.zero_grad()
                return
            for p, l in zip(mlp.params, range(1, mlp.L+1)):
                mlp.params[p][l] = GD_update(
                    mlp.params[p][l], mlp.grads[p][l] / batch_size, alpha
                )
            losses[k].append(round(b_train_loss, 4))
        print(f"Iter {k}: loss = {b_train_loss}")
    return losses    

