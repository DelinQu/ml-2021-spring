import torch
from model import Classifier

'''
traing on the tt_set and validate on val_set
'''
def train(tr_set, val_set, model:Classifier, config:dict, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    best_acc, early_stop_cnt, epoch, n_epochs = 0., 0, 0, config['n_epochs']
    acc_record = {
        'train':[],
        'val':[]
    }

    while epoch < n_epochs:
        # train each batch
        epoch_acc, epoch_loss = 0.0, 0.0
        model.train()
        for x, y in tr_set:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            batch_loss = model.cal_loss(pred, y)
            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item()
            pred = torch.max(pred,1)
            epoch_acc += (pred.cpu() == y.cpu).sum().item()

        # validation
        val_loss, val_acc = calc_val_loss(val_set, model, device)
        epoch_loss /= len(tr_set)
        epoch_acc /= len(tr_set.dataset)
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
            epoch + 1, epoch, epoch_acc / len(tr_set), epoch_loss / len(tr_set), val_acc / len(val_set),
            val_loss / len(val_set)
        ))

        # save model
        if val_acc > best_acc:
            best_acc = val_acc
            print('Saving model (epoch = {:4d}, accuracy = {:.4f})'.format(epoch + 1, best_acc))
            torch.save(model.state_dict(),config['save_path'])
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        acc_record['train'].append(epoch_acc)
        acc_record['val'].append(val_acc)

        # stop training if model stops improving for early_stop epochs.
        if early_stop_cnt > config['early_stop']:
            break
        epoch += 1

    print('Finished training after {} epochs'.format(epoch))
    return best_acc, acc_record


'''
caculate total loss in val_set.
'''
def calc_val_loss(val_set, model:Classifier, device):
    model.eval()                                    # set model to evalutation mode
    total_loss, total_acc = 0.0, 0.0
    for x, y in val_set:                            # iterate through the dataloader
        x, y = x.to(device), y.to(device)           # move data to valice (cpu/cuda)
        with torch.no_grad():                       # disable gradient calculation
            pre = model(x)                          # forward pass (compute output)
            batch_loss = model.cal_loss(pre, y)     # compute loss

        total_loss += batch_loss.item()
        total_acc += (pre.cpu() == y.cpu()).sum().item()

    total_loss /= len(val_set)                      # compute averaged loss by div len dataloader
    total_acc /= len(val_set.dataset)               # compute averaged acc by div len dataset
    return total_loss


'''
predict on val set, return preds and targets for plot
'''
def pred_val(dv_set, model, device):
    model.eval()
    preds, targets = [], []
    for x, y in dv_set:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x)
            pred = torch.max(pred, 1)
            preds.append(pred.detach().cpu())
            targets.append(y.detach().cpu())

    preds = torch.cat(preds, dim=0).numpy()
    targets = torch.cat(targets, dim=0).numpy()

    return preds, targets