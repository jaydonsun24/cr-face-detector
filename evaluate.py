import torch
import matplotlib.pyplot as plt
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from data_setup import class_names
from data_setup import test_dataloader, device, test_data
from tqdm.auto import tqdm
from model import model

model.load_state_dict(torch.load("models/emotion_model.pth"))


#1. Make predictions
y_preds = [ ]
model.eval()
with torch.inference_mode():
  for X, y in tqdm(test_dataloader, desc = "Making predictions ..."):
    X, y = X.to(device), y.to(device)

    #fwd pass
    y_logit = model(X)

    #turn to pred probs and pred labels
    y_pred = torch.argmax(y_logit, dim = 1)

    #Put prediction on CPU for evaluation
    y_preds.append(y_pred.cpu())

  #Concatenate list of predictions into a tensor
  # print(y_preds)
y_pred_tensor = torch.cat(y_preds)

confmat = ConfusionMatrix(task = "multiclass", num_classes=len(class_names))

confmat_tensor = confmat(preds = y_pred_tensor,
                         target = torch.tensor(test_data.targets))

fig, ax = plot_confusion_matrix(
    conf_mat = confmat_tensor.numpy(), #matplot lib likes numpy
    class_names = class_names,
    figsize = (10,7)
)

plt.show()