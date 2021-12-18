import pandas as pd
import torch
from model import NeuralNet

'''
predict with testing set
'''
def test(tt_set, model:NeuralNet, device):
    model.eval()                                # set model to evalutation mode
    preds = []
    for x in tt_set:                            # iterate through the dataloader
        x = x.to(device)                        # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            preds.append(pred.detach().cpu())   # collect prediction
    preds = torch.cat(preds, dim=0).numpy()     # concatenate all predictions and convert to a numpy array
    return preds

''' 
Save predictions to specified file 
'''
def save_pred(preds, file):
    print('Saving results to {}'.format(file))
    df = pd.DataFrame({
        'id': list(range(preds.shape[0])),
        'tested_positive': preds.tolist()
    })
    df.to_csv(file, index=False)