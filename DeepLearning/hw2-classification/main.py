import os
import torch

from TIMITDataset import TIMITDataLoader
from model import Classifier
from testing import test, save_pred
from training import train, pred_val
from utils import same_seeds, get_device, plot_learning_curve, plot_dev_pred

config = {
    'n_epochs': 20,                     # maximum number of epochs
    'batch_size': 200,                  # mini-batch size for dataloader
    'lr': 0.001,                        # learning rate
    'early_stop': 500,                  # early stopping epochs (the number epochs since your model's last improvement)
    'save_path': 'models/model.ckpt',   # your model will be saved here
    'tr_path': 'covid.train.csv',       # path to training data
    'tt_path': 'covid.test.csv',        # path to testing data
    'target_only': False
}


same_seeds(0)
device = get_device()
os.makedirs('models',exist_ok=True)

if __name__ == '__main__':
    # create dataloader
    train_set = TIMITDataLoader('train', config['batch_size'])
    val_set = TIMITDataLoader('val', config['batch_size'])
    test_set = TIMITDataLoader('test',config['batch_size'])

    # create model
    model = Classifier().to(device)
    # training and plot training loss curve
    best_acc, acc_record = train(train_set, val_set, model, config, device)
    plot_learning_curve(acc_record, title='Classifier model')

    # load model and plot prediction curve on dev
    del model
    model = Classifier().to(device)
    ckpt = torch.load(config['save_path'], map_location='cpu')  # Load your best model
    model.load_state_dict(ckpt)
    preds, targets =  pred_val(val_set, model, device)           # predct on dev_set
    plot_dev_pred(preds, targets)                               # Show prediction on the validation set

    # predict on the testing set.
    preds = test(test_set, model, device)
    save_pred(preds, 'pred.csv')



