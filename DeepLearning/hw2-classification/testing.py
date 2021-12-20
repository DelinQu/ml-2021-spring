import pandas as pd
import torch
from model import Classifier

'''
predict with testing set
'''
def test(test_set, model:Classifier, device):
    model.eval()                                # set model to evalutation mode
    preds = []
    for x in test_set:                          # iterate through the dataloader
        x = x.to(device)                        # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            pred = torch.max(pred,1)
            preds.append(pred.detach().cpu())   # collect prediction
    preds = torch.cat(preds, dim=0).numpy()     # concatenate all predictions and convert to a numpy array
    return preds

''' 
Save predictions to specified file 
'''
def save_pred(preds, file):
    print('Saving results to {}'.format(file))
    df = pd.DataFrame({
        'Id': list(range(preds.shape[0])),
        'Class': preds.tolist()
    })
    df.to_csv(file, index=False)