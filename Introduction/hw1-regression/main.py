import os
import torch

from COVID19Dataset import COVID19Dataloader
from model import NeuralNet
from training import train, pred_dev
from utils import get_device, plot_learning_curve, plot_dev_pred, same_seeds
from testing import test, save_pred

# TODO: How to tune these hyper-parameters to improve your model's performance?
config = {
    'n_epochs': 10000,                  # maximum number of epochs
    'batch_size': 200,                  # mini-batch size for dataloader
    'optimizer': 'Adam',                # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {                   # hyper-parameters for the optimizer (depends on which optimizer you are using)
        # 'lr': 0.001,                    # learning rate of SGD
        # 'momentum': 0.9                 # momentum for SGD
        # 'weight_decay': 5e-4,
    },
    'lambda': 0.00075,                  # regularization rate
    'early_stop': 500,                  # early stopping epochs (the number epochs since your model's last improvement)
    'save_path': 'models/model.pth',    # your model will be saved here
    'tr_path': 'covid.train.csv',       # path to training data
    'tt_path': 'covid.test.csv',        # path to testing data
    'target_only': False
}

same_seeds(0)
device = get_device()
os.makedirs('models',exist_ok=True)
target_only = True

if __name__ == '__main__':
    # get dataloader
    tr_set = COVID19Dataloader(config['tr_path'], 'train', config['batch_size'], target_only = config['target_only'])
    dv_set = COVID19Dataloader(config['tr_path'], 'dev',   config['batch_size'], target_only = config['target_only'])
    tt_set = COVID19Dataloader(config['tt_path'], 'test', config['batch_size'], target_only = config['target_only'])

    # construct model and move to device
    model = NeuralNet(tr_set.dataset.dim, config['lambda']).to(device)

    # training and plot training loss curve
    model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)
    plot_learning_curve(model_loss_record, title='deep model')

    # load model and plot prediction curve on dev
    del model
    model = NeuralNet(tr_set.dataset.dim, config['lambda']).to(device)
    ckpt = torch.load(config['save_path'], map_location='cpu')  # Load your best model
    model.load_state_dict(ckpt)
    preds, targets =  pred_dev(dv_set, model, device)           # predct on dev_set
    plot_dev_pred(preds, targets)                               # Show prediction on the validation set

    # predict on the testing set.
    preds = test(tt_set, model, device)
    save_pred(preds, 'pred.csv')
