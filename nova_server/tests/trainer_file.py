from sklearn.svm import SVC
from logging import CRITICAL, WARNING, INFO, DEBUG
logger = None

def log(msg, ll):
    if logger:
        logger.log(ll, msg)
    else:
        print(msg)

def train_with_log(x, y, l=None):
    logger = l
    train(x, y)

def train(x, y):

    svc = SVC()
    log('Fitting SVM to data...', INFO)
    svc.fit(x,y)
    log('...done', INFO)
    return
