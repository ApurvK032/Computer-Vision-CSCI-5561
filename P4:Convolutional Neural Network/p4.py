import numpy as np
import matplotlib.pyplot as plt
import main_functions as main


def get_mini_batch(im_train, label_train, batch_size):
    mini_batch_x, mini_batch_y = [], []
    
    # TO DO
    num_samples = im_train.shape[1]
    indices = np.random.permutation(num_samples)
    
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_x = im_train[:, batch_indices]
        batch_labels = label_train[:, batch_indices]
        
        batch_size_actual = len(batch_indices)
        batch_y = np.zeros((10, batch_size_actual))
        for j in range(batch_size_actual):
            batch_y[batch_labels[0, j], j] = 1
        
        mini_batch_x.append(batch_x)
        mini_batch_y.append(batch_y)
                    
    return mini_batch_x, mini_batch_y


def fc(x, w, b):
    y = None

    # TO DO
    y = np.dot(w, x) + b
    
    return y


def fc_backward(dl_dy, x, w, b, y):
    dl_dx, dl_dw, dl_db = None, None, None

    # TO DO
    dl_dx = np.dot(w.T, dl_dy)
    dl_dw = np.dot(dl_dy, x.T)
    dl_db = dl_dy

    return dl_dx, dl_dw, dl_db


def loss_euclidean(y_tilde, y):
    l, dl_dy = None, None

    # TO DO
    diff = y_tilde - y
    l = np.sum(diff ** 2)
    dl_dy = 2 * diff

    return l, dl_dy

def loss_cross_entropy_softmax(a, y):
    l, dl_da = None, None

    # TO DO
    a_max = np.max(a, axis=0, keepdims=True)
    exp_a = np.exp(a - a_max)
    y_tilde = exp_a / np.sum(exp_a, axis=0, keepdims=True)
    
    l = -np.sum(y * np.log(y_tilde + 1e-10))
    
    dl_da = y_tilde - y

    return l, dl_da

def relu(x):
    y = None

    # TO DO
    y = np.maximum(0, x)

    return y


def relu_backward(dl_dy, x, y):
    dl_dx = None

    # TO DO
    dl_dx = dl_dy * (x > 0)

    return dl_dx


def conv(x, w_conv, b_conv):
    y = None

    # TO DO
    H, W, C1 = x.shape
    h, w, _, C2 = w_conv.shape
    pad = h // 2
    
    x_padded = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
    y = np.zeros((H, W, C2))
    
    w_reshaped = w_conv.reshape(-1, C2)
    
    for i in range(H):
        for j in range(W):
            region = x_padded[i:i+h, j:j+w, :].reshape(-1)
            y[i, j, :] = region.dot(w_reshaped) + b_conv[:, 0]
 
    return y


def conv_backward(dl_dy, x, w_conv, b_conv, y):
    dl_dw, dl_db = None, None

    # TO DO
    H, W, C1 = x.shape
    h, w, _, C2 = w_conv.shape
    pad = h // 2
    
    x_padded = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
    dl_dw = np.zeros_like(w_conv)
    dl_db = np.zeros_like(b_conv)
    
    for i in range(H):
        for j in range(W):
            region = x_padded[i:i+h, j:j+w, :]
            for c2 in range(C2):
                dl_dw[:, :, :, c2] += region * dl_dy[i, j, c2]
    
    dl_db[:, 0] = np.sum(dl_dy.reshape(H*W, C2), axis=0)
    
    return dl_dw, dl_db

def pool2x2(x):
    y = None

    # TO DO
    H, W, C = x.shape
    y = np.zeros((H // 2, W // 2, C))
    
    for c in range(C):
        for i in range(H // 2):
            for j in range(W // 2):
                y[i, j, c] = np.max(x[i*2:i*2+2, j*2:j*2+2, c])

    return y

def pool2x2_backward(dl_dy, x, y):
   dl_dx = None
   
   # TO DO
   H, W, C = x.shape
   dl_dx = np.zeros_like(x)
   
   for c in range(C):
       for i in range(H // 2):
           for j in range(W // 2):
               region = x[i*2:i*2+2, j*2:j*2+2, c]
               max_val = np.max(region)
               mask = (region == max_val)
               dl_dx[i*2:i*2+2, j*2:j*2+2, c] += mask * dl_dy[i, j, c]

   return dl_dx


def flattening(x):
    y = None

    # TO DO
    y = x.reshape((-1, 1))

    return y


def flattening_backward(dl_dy, x, y):
    dl_dx = None

    # TO DO
    dl_dx = dl_dy.reshape(x.shape)

    return dl_dx


def train_slp_linear(mini_batch_x, mini_batch_y):
    w, b = None, None

    # TO DO
    learning_rate = 0.01
    decay_rate = 0.9
    num_iterations = 8000
    
    w = np.random.randn(10, 196)
    b = np.random.randn(10, 1)
    
    num_batches = len(mini_batch_x)
    k = 0
    
    for iteration in range(num_iterations):
        if iteration % 1000 == 0 and iteration > 0:
            learning_rate *= decay_rate
        
        batch_x = mini_batch_x[k]
        batch_y = mini_batch_y[k]
        batch_size = batch_x.shape[1]
        
        dl_dw_total = np.zeros_like(w)
        dl_db_total = np.zeros_like(b)
        
        for i in range(batch_size):
            x = batch_x[:, [i]]
            y_true = batch_y[:, [i]]
            
            y_pred = fc(x, w, b)
            loss, dl_dy = loss_euclidean(y_pred, y_true)
            dl_dx, dl_dw, dl_db = fc_backward(dl_dy, x, w, b, y_pred)
            
            dl_dw_total += dl_dw
            dl_db_total += dl_db
        
        w -= (learning_rate / batch_size) * dl_dw_total
        b -= (learning_rate / batch_size) * dl_db_total
        
        k = (k + 1) % num_batches
    
    return w, b

def train_slp(mini_batch_x, mini_batch_y):
    w, b = None, None

    # TO DO
    learning_rate = 0.1
    decay_rate = 0.9
    num_iterations = 5000
    
    w = np.random.randn(10, 196)
    b = np.random.randn(10, 1)
    
    num_batches = len(mini_batch_x)
    k = 0
    
    for iteration in range(num_iterations):
        if iteration % 1000 == 0 and iteration > 0:
            learning_rate *= decay_rate
        
        batch_x = mini_batch_x[k]
        batch_y = mini_batch_y[k]
        batch_size = batch_x.shape[1]
        
        dl_dw_total = np.zeros_like(w)
        dl_db_total = np.zeros_like(b)
        
        for i in range(batch_size):
            x = batch_x[:, [i]]
            y_true = batch_y[:, [i]]
            
            a = fc(x, w, b)
            loss, dl_da = loss_cross_entropy_softmax(a, y_true)
            dl_dx, dl_dw, dl_db = fc_backward(dl_da, x, w, b, a)
            
            dl_dw_total += dl_dw
            dl_db_total += dl_db
        
        w -= (learning_rate / batch_size) * dl_dw_total
        b -= (learning_rate / batch_size) * dl_db_total
        
        k = (k + 1) % num_batches
    
    return w, b

def train_mlp(mini_batch_x, mini_batch_y):
    w1, b1, w2, b2 = None, None, None, None

    # TO DO
    learning_rate = 0.2
    decay_rate = 0.9
    num_iterations = 10000
    
    w1 = np.random.randn(30, 196) * 0.1
    b1 = np.zeros((30, 1))
    w2 = np.random.randn(10, 30) * 0.1
    b2 = np.zeros((10, 1))
    
    num_batches = len(mini_batch_x)
    k = 0
    
    for iteration in range(num_iterations):
        if iteration % 1000 == 0 and iteration > 0:
            learning_rate *= decay_rate
        
        batch_x = mini_batch_x[k]
        batch_y = mini_batch_y[k]
        batch_size = batch_x.shape[1]
        
        dl_dw1_total = np.zeros_like(w1)
        dl_db1_total = np.zeros_like(b1)
        dl_dw2_total = np.zeros_like(w2)
        dl_db2_total = np.zeros_like(b2)
        
        for i in range(batch_size):
            x = batch_x[:, [i]]
            y_true = batch_y[:, [i]]
            
            h1 = fc(x, w1, b1)
            h2 = relu(h1)
            a = fc(h2, w2, b2)
            
            loss, dl_da = loss_cross_entropy_softmax(a, y_true)
            
            dl_dh2, dl_dw2, dl_db2 = fc_backward(dl_da, h2, w2, b2, a)
            dl_dh1 = relu_backward(dl_dh2, h1, h2)
            dl_dx, dl_dw1, dl_db1 = fc_backward(dl_dh1, x, w1, b1, h1)
            
            dl_dw1_total += dl_dw1
            dl_db1_total += dl_db1
            dl_dw2_total += dl_dw2
            dl_db2_total += dl_db2
        
        w1 -= (learning_rate / batch_size) * dl_dw1_total
        b1 -= (learning_rate / batch_size) * dl_db1_total
        w2 -= (learning_rate / batch_size) * dl_dw2_total
        b2 -= (learning_rate / batch_size) * dl_db2_total
        
        k = (k + 1) % num_batches

    return w1, b1, w2, b2


def train_cnn(mini_batch_x, mini_batch_y):
    w_conv, b_conv, w_fc, b_fc = None, None, None, None

    # TO DO
    learning_rate = 0.1
    decay_rate = 0.9
    num_iterations = 10000
    
    w_conv = np.random.randn(3, 3, 1, 3) * 0.1
    b_conv = np.zeros((3, 1))
    w_fc = np.random.randn(10, 147) * 0.1
    b_fc = np.zeros((10, 1))
    
    num_batches = len(mini_batch_x)
    k = 0
    
    for iteration in range(num_iterations):
        if iteration % 1000 == 0 and iteration > 0:
            learning_rate *= decay_rate
        
        batch_x = mini_batch_x[k]
        batch_y = mini_batch_y[k]
        batch_size = batch_x.shape[1]
        
        dl_dw_conv_total = np.zeros_like(w_conv)
        dl_db_conv_total = np.zeros_like(b_conv)
        dl_dw_fc_total = np.zeros_like(w_fc)
        dl_db_fc_total = np.zeros_like(b_fc)
        
        for i in range(batch_size):
            x = batch_x[:, [i]].reshape((14, 14, 1), order='F')
            y_true = batch_y[:, [i]]
            
            conv_out = conv(x, w_conv, b_conv)
            relu_out = relu(conv_out)
            pool_out = pool2x2(relu_out)
            flat_out = flattening(pool_out)
            a = fc(flat_out, w_fc, b_fc)
            
            loss, dl_da = loss_cross_entropy_softmax(a, y_true)
            
            dl_dflat, dl_dw_fc, dl_db_fc = fc_backward(dl_da, flat_out, w_fc, b_fc, a)
            dl_dpool = flattening_backward(dl_dflat, pool_out, flat_out)
            dl_drelu = pool2x2_backward(dl_dpool, relu_out, pool_out)
            dl_dconv = relu_backward(dl_drelu, conv_out, relu_out)
            dl_dw_conv, dl_db_conv = conv_backward(dl_dconv, x, w_conv, b_conv, conv_out)
            
            dl_dw_conv_total += dl_dw_conv
            dl_db_conv_total += dl_db_conv
            dl_dw_fc_total += dl_dw_fc
            dl_db_fc_total += dl_db_fc
        
        w_conv -= (learning_rate / batch_size) * dl_dw_conv_total
        b_conv -= (learning_rate / batch_size) * dl_db_conv_total
        w_fc -= (learning_rate / batch_size) * dl_dw_fc_total
        b_fc -= (learning_rate / batch_size) * dl_db_fc_total
        
        k = (k + 1) % num_batches

    return w_conv, b_conv, w_fc, b_fc


if __name__ == '__main__':
    main.main_slp_linear(load_weights=False)
    main.main_slp(load_weights=False)
    main.main_mlp(load_weights=False)
    main.main_cnn(load_weights=False)