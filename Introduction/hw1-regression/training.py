import torch
from model import NeuralNet

'''
traing on the tt_set and validate on dev_set
'''
def train(tr_set, dv_set, model:NeuralNet, config:dict, device):
    optimizer = getattr(torch.optim, config.get('optimizer'))\
                (model.parameters(), **config['optim_hparas'])

    min_loss, early_stop_cnt, epoch, n_epochs = 1000., 0, 0, config['n_epochs']
    loss_record = {
        'train':[],
        'dev':[]
    }

    while epoch < n_epochs:
        model.train()                               # set model to training mode
        for x, y in tr_set:                         # iterate through the dataloader
            optimizer.zero_grad()                   # set gradient to zero
            x, y = x.to(device), y.to(device)       # move data to device (cpu/cuda)
            pred = model(x)                         # forward pass (compute output)
            loss = model.cal_loss(pred, y)          # compute loss
            loss.backward()                         # compute gradient (backpropagation)
            optimizer.step()                        # update model with optimizer
            loss_record['train'].append(loss.detach().cpu().item())

            # test model on validation (development) set after a epoch.
            dev_loss = calc_dev_loss(dv_set, model, device)
            if dev_loss < min_loss:
                min_loss = dev_loss
                print('Saving model (epoch = {:4d}, loss = {:.4f})'.format(epoch + 1, min_loss))
                torch.save(model.state_dict(),config['save_path'])
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1

            epoch += 1
            loss_record['dev'].append(dev_loss)
            # stop training if model stops improving for early_stop epochs.
            if early_stop_cnt > config['early_stop']:
                break

    print('Finished training after {} epochs'.format(epoch))
    return min_loss, loss_record


'''
caculate total loss in dev_set.
'''
def calc_dev_loss(dv_set, model:NeuralNet, device):
    model.eval()                                    # set model to evalutation mode
    total_loss = 0
    for x, y in dv_set:                             # iterate through the dataloader
        x, y = x.to(device), y.to(device)           # move data to device (cpu/cuda)
        with torch.no_grad():                       # disable gradient calculation
            pre = model(x)                          # forward pass (compute output)
            loss = model.cal_loss(pre,y)            # compute loss
        total_loss += loss.detach().cpu().item() * len(x)
    total_loss /= len(dv_set.dataset)               # compute averaged loss

    return total_loss


'''
predict on dev set, return preds and targets for plot
'''
def pred_dev(dv_set, model, device):
    model.eval()
    preds, targets = [], []
    for x, y in dv_set:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
            targets.append(y.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    targets = torch.cat(targets, dim=0).numpy()

    return preds, targets