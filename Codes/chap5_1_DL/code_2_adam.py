from copy import deepcopy
import numpy as np
from numpy import ndarray as array
from tqdm import tqdm


class MLP_mv:
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
        self.vs = deepcopy(self.grads)  
        self.ms = deepcopy(self.grads)  

    def backward_with_mv(self, dLdy, beta_1=0.9, beta_2=0.9):
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
            self.ms["b"][l] = beta_1 * self.ms["b"][l] + (1-beta_1) * self.grads["b"][l]
            self.ms["W"][l] = beta_1 * self.ms["W"][l] + (1-beta_1) * self.grads["W"][l]
            self.vs["b"][l] = beta_2 * self.vs["b"][l] + (1-beta_2) * self.grads["b"][l] ** 2
            self.vs["W"][l] = beta_2 * self.vs["W"][l] + (1-beta_2) * self.grads["W"][l] ** 2
        pass




def RMSProp_opt_MLP(L_func, dL_func, mlp:MLP_mv, xs:array, ys:array, loss_target, batch_size,  
    N_I:int=100, rho=0.9, alpha_0=1e-2, gamma=1-1e-2, epsilon=1e-6):
    def RMSProp_update(param, grad, v, alpha):
        return param - alpha * grad / np.sqrt(v + epsilon) # <-- different here
    N, N_b = xs.shape[0], xs.shape[0] // batch_size
    alpha, losses = alpha_0, [] # <-- different here
    assert ys.shape[0] == N
    for k in range(N_I):
        ids = np.arange(N)
        np.random.shuffle(ids)
        for b in tqdm(range(N_b)):
            mlp.zero_grad()
            b_ids = ids[b*batch_size:(b+1)*batch_size]
            b_train_loss, b_xs, b_ys  = 0, xs[b_ids], ys[b_ids]
            for i in range(batch_size):
                y_est = mlp.forward(b_xs[i])
                b_train_loss += L_func(b_ys[i], y_est)
                dLdy = dL_func(b_ys[i], y_est)
                mlp.backward_with_mv(dLdy, 0, rho) # <-- different here
            b_train_loss = b_train_loss / batch_size
            if b_train_loss < loss_target:
                print(f"Iter {k} batch loss = {b_train_loss:.4f} reach target")
                mlp.zero_grad()
                return
            for p, l in zip(mlp.params, range(1, mlp.L+1)):
                mlp.params[p][l] = RMSProp_update( # <-- different here
                    mlp.params[p][l], mlp.grads[p][l] / batch_size, mlp.vs[p][l], alpha
                )   
            losses.append((k, b_train_loss))
        alpha = alpha * gamma # <-- different here
        print(f"Iter {k}: loss = {b_train_loss}")
    return losses    



def Momentum_opt_MLP(L_func, dL_func, mlp:MLP_mv, xs:array, ys:array, loss_target, batch_size,  
    N_I:int=100, beta=0.9, alpha_0=1e-2, gamma=1-1e-2, epsilon=1e-6):
    def Momentum_update(param, momentum, alpha):
        return param - alpha * momentum # <-- different here
    N, N_b = xs.shape[0], xs.shape[0] // batch_size
    alpha, losses = alpha_0, [] # <-- different here
    assert ys.shape[0] == N
    for k in range(N_I):
        ids = np.arange(N)
        np.random.shuffle(ids)
        for b in tqdm(range(N_b)):
            mlp.zero_grad()
            b_ids = ids[b*batch_size:(b+1)*batch_size]
            b_train_loss, b_xs, b_ys  = 0, xs[b_ids], ys[b_ids]
            for i in range(batch_size):
                y_est = mlp.forward(b_xs[i])
                b_train_loss += L_func(b_ys[i], y_est)
                dLdy = dL_func(b_ys[i], y_est)
                mlp.backward_with_mv(dLdy, beta, 0) # <-- different here
            b_train_loss = b_train_loss / batch_size
            if b_train_loss < loss_target:
                print(f"Iter {k} batch loss = {b_train_loss:.4f} reach target")
                mlp.zero_grad()
                return
            for p, l in zip(mlp.params, range(1, mlp.L+1)):
                mlp.params[p][l] = Momentum_update( # <-- different here
                    mlp.params[p][l], mlp.ms[p][l], alpha
                )
            losses.append((k, b_train_loss))
        alpha = alpha * gamma # <-- different here
        print(f"Iter {k}: loss = {b_train_loss}")
    return losses    




def Adam_opt_MLP(L_func, dL_func, mlp:MLP_mv, xs:array, ys:array, loss_target, batch_size,  
    N_I:int=100, beta_1=0.9, beta_2=0.9, alpha_0=1e-2, gamma=1-1e-2, epsilon=1e-6):
    def Adam_update(param, m, v, alpha):
        return param - alpha * m / np.sqrt(v + epsilon) # <-- different here
    N, N_b = xs.shape[0], xs.shape[0] // batch_size
    alpha, losses = alpha_0, [] # <-- different here
    assert ys.shape[0] == N
    for k in range(N_I):
        ids = np.arange(N)
        np.random.shuffle(ids)
        for b in tqdm(range(N_b)):
            mlp.zero_grad()
            b_ids = ids[b*batch_size:(b+1)*batch_size]
            b_train_loss, b_xs, b_ys  = 0, xs[b_ids], ys[b_ids]
            for i in range(batch_size):
                y_est = mlp.forward(b_xs[i])
                b_train_loss += L_func(b_ys[i], y_est)
                dLdy = dL_func(b_ys[i], y_est)
                mlp.backward_with_mv(dLdy, beta_1, beta_2) # <-- different here
            b_train_loss = b_train_loss / batch_size
            if b_train_loss < loss_target:
                mlp.zero_grad()
                print(f"Iter {k} batch loss = {b_train_loss:.4f} reach target")
                return
            for p, l in zip(mlp.params, range(1, mlp.L+1)):
                mlp.params[p][l] = Adam_update( # <-- different here
                    mlp.params[p][l], mlp.ms[p][l], mlp.vs[p][l], alpha
                )
            losses.append((k, b_train_loss))
        alpha = alpha * gamma # <-- different here
        print(f"Iter {k}: loss = {b_train_loss}")
    pass   

