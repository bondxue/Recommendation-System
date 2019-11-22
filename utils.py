import var
import numpy as np

def cal_sparsity():
    sparsity = float(len(ratings.nonzero()[0]))
    sparsity /= (ratings.shape[0] * ratings.shape[1])
    sparsity *= 100
    print('Matrix density of training set is: {:4.2f}%'.format(sparsity))
    
def cal_mean():
    '''Calculate mean value'''
    # population mean, each uesr mean, each item mean 
    global all_mean, user_mean, item_mean 
    all_mean = np.mean(ratings[ratings!=0])
    user_mean = sum(ratings.T) / sum((ratings!=0).T)
    item_mean = sum(ratings) / sum((ratings!=0))
    print('Exist User/Item mean NaN?', np.isnan(user_mean).any(), np.isnan(item_mean).any())
    # fill in NaN with population mean
    user_mean = np.where(np.isnan(user_mean), all_mean, user_mean)
    item_mean = np.where(np.isnan(item_mean), all_mean, item_mean)
    print('Exist User/Item mean NaN?', np.isnan(user_mean).any(), np.isnan(item_mean).any())
    print('Finsh，population mean is %.4f' % all_mean)
    
def cal_similarity(ratings, kind, epsilon=1e-9):
    '''uisng Cosine distance to calculate similarilty'''
    '''epsilon: aviod Divide-by-zero error ，Correct it.'''
    if kind == 'user':
        sim = ratings.dot(ratings.T) + epsilon
    elif kind == 'item':
        sim = ratings.T.dot(ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)

def cal_similarity_norm(ratings, kind, epsilon=1e-9):
    '''Normalized index:Pearson correlation coefficient'''
    if kind == 'user':
        # normalize the ratings of same user 
        rating_user_diff = ratings.copy()
        for i in range(ratings.shape[0]):
            nzero = ratings[i].nonzero()
            rating_user_diff[i][nzero] = ratings[i][nzero] - user_mean[i]
        sim = rating_user_diff.dot(rating_user_diff.T) + epsilon
    elif kind == 'item':
        # normalized the ratings of same item 
        rating_item_diff = ratings.copy()
        for j in range(ratings.shape[1]):
            nzero = ratings[:,j].nonzero()
            rating_item_diff[:,j][nzero] = ratings[:,j][nzero] - item_mean[j]
        sim = rating_item_diff.T.dot(rating_item_diff) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)

def rmse(pred, actual):
    '''calculate prediction rmse'''
    from sklearn.metrics import mean_squared_error
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return np.sqrt(mean_squared_error(pred, actual))