import torch
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np

torch.manual_seed(1)

class CustomOneCycleLR():
    """Custom class for one cycle lr
    
    """
    def __init__(self, optimizer, schedule, steps_per_epoch):
        self.optimizer = optimizer
        self.schedule = schedule
        self.epoch = 0
        self.steps = 0
        self.steps_per_epoch = steps_per_epoch
        self.optimizer.param_groups[0]['lr'] = self.schedule[self.epoch]
        
    def step(self):
        self.optimizer.param_groups[0]['lr'] = self.lr_schedules()
    
    def lr_schedules(self):
        self.steps += 1
        if self.steps%self.steps_per_epoch == 0:
            self.steps = 1
            self.epoch += 1
        return self.schedule[self.epoch]

def one_cycle_lr_pt(optimizer, lr, max_lr, steps_per_epoch, epochs, anneal_strategy='linear'):
    return OneCycleLR(
        optimizer, 
        max_lr=max_lr, 
        steps_per_epoch=steps_per_epoch, 
        epochs=epochs, 
        anneal_strategy='linear'
    )

def one_cycle_lr_custom(optimizer, lr, max_lr, steps_per_epoch, epochs, anneal_strategy='linear'):
    if epochs < 12:
        raise Exception("Epoch value can not be less than 12")
    schedule = np.interp(np.arange(epochs+1), [0, 2, 8, 12, epochs], [lr, max_lr, lr/5.0, lr/20.0, 0])
    return CustomOneCycleLR(optimizer, schedule, steps_per_epoch)