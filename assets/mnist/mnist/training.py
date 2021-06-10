import time
import math
import torch

torch.manual_seed(1)

class Training():
    def __init__(self, 
                 model,
                 optimizer,
                 scheduler,
                 train,
                 test,
                 train_loader,
                 test_loader,
                 lr,
                 epochs,
                 device,
                 dropout
            ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train = train
        self.test = test
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.dropout = dropout
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        self.list_train_loss = []
        self.list_valid_loss = []
        self.list_train_correct = []
        self.list_valid_correct = []
        
        self.schedule = []
        
        self.start_time = 0
        self.end_time = 0
        
        self.best_perc = 99.2
        self.best_path = ""
    
    def epoch_time(self):
        elapsed_time = self.end_time - self.start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
    
    def print_epoch_progress(self, train_correct, train_loss, valid_correct, valid_loss):
        epoch_mins, epoch_secs = self.epoch_time()
        print(f'\t          Time: {epoch_mins}m {epoch_secs}s');
        print(f'\t    Train Loss: {train_loss:.6f}')
        print(f'\tTrain Accuracy: {train_correct:5d}/{len(self.train_loader.dataset):5d} | Percent: {(100. * train_correct / len(self.train_loader.dataset)):.2f}%')
        print(f'\t     Val. Loss: {valid_loss:.6f}')
        print(f'\t  Val Accuracy: {valid_correct:5d}/{len(self.test_loader.dataset):5d} | Percent: {(100. * valid_correct / len(self.test_loader.dataset)):.2f}%')
    
    
    def log_epoch_params(self, epoch):
        print(f'Epoch: {epoch+1:02}')
        print(f'\t Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
    
    def save_best(self, valid_correct):
        valid_perc = (100. * valid_correct / len(self.test_loader.dataset))

        if valid_perc >= self.best_perc:
            self.best_perc = valid_perc
            self.best_path = f'model_weights_{valid_perc:.2f}.pth'
            torch.save(self.model.state_dict(), self.best_path)
    
    def run(self):
        for epoch in range(self.epochs):
            self.schedule.append(self.optimizer.param_groups[0]['lr'])
            self.log_epoch_params(epoch)
            
            self.start_time = time.time()

            train_loss, train_correct = self.train(self.model, self.train_loader, self.optimizer, self.dropout, self.device, self.scheduler)
            valid_loss, valid_correct = self.test(self.model, self.test_loader, self.device)

            self.list_train_loss.append(train_loss)
            self.list_valid_loss.append(valid_loss)

            self.list_train_correct.append(100. * train_correct / len(self.train_loader.dataset))
            self.list_valid_correct.append(100. * valid_correct / len(self.test_loader.dataset))

            self.end_time = time.time()
            
            self.save_best(valid_correct)

            self.print_epoch_progress(train_correct, train_loss, valid_correct, valid_loss)
            
            
    def print_best_model(self):
        self.model.load_state_dict(torch.load(self.best_path))
        self.model.eval()

        valid_loss, valid_correct = self.test(self.model, self.test_loader, self.device)

        print(f'Val Accuracy: {valid_correct:4d}/{len(self.test_loader.dataset):5d}')
        print(f'     Percent: {(100. * valid_correct / len(self.test_loader.dataset)):.2f}%')
        print(f'   Val. Loss: {valid_loss:.6f}')