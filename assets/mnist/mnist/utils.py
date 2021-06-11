import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

torch.manual_seed(1)

def get_device() -> tuple:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return (use_cuda, device)


def print_summary(model, input_size=(1, 28, 28)):
    summary(model, input_size=input_size)


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
        

def load_weights_from_path(model, path):
    model.load_state_dict(torch.load(path))
    return model


def get_all_predictions(model, loader, device):
    model.eval()
    all_preds = torch.tensor([]).to(device)
    all_targets = torch.tensor([]).to(device)
    with torch.no_grad():
        for data, target in loader:
            data, targets = data.to(device), target.to(device)
            all_targets = torch.cat(
                (all_targets, targets),
                dim=0
            )
            output = model(data)
            preds = output.argmax(dim=1)
            all_preds = torch.cat(
                (all_preds, preds),
                dim=0
            ) 
    
    return all_preds, all_targets


def get_incorrrect_predictions(model, loader, device):
    model.eval()
    incorrect = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)
            pred = output.argmax(dim=1)
            for d, t, p, o in zip(data, target, pred, output):
                if p.eq(t.view_as(p)).item() == False:
                    incorrect.append([d.cpu(), t.cpu(), p.cpu(), o[p.item()].cpu()])
    
    return incorrect


def prepare_confusion_matrix(all_preds, all_targets, class_map):
    stacked = torch.stack((
            all_targets, all_preds
        ),
        dim=1
    ).type(torch.int64)

    no_classes = len(class_map)

    # Create temp confusion matrix
    confusion_matrix = torch.zeros(no_classes, no_classes, dtype=torch.int64)

    # Fill up confusion matrix with actual values
    for p in stacked:
        tl, pl = p.tolist()
        confusion_matrix[tl, pl] = confusion_matrix[tl, pl] + 1

    return confusion_matrix