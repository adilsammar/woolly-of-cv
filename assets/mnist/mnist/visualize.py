import matplotlib.pyplot as plt
import numpy as np
import itertools


def print_samples(train_loader, count=16):
    # Print Random Samples
    if not count%8 == 0:
        return
    fig = plt.figure(figsize = (15, 5))
    for imgs, labels in train_loader:
        for i in range(count):
            ax = fig.add_subplot(int(count/8), 8, i + 1, xticks = [], yticks = [])
            ax.set_title(f'Label: {labels[i]}')
            plt.imshow(imgs[i].numpy().transpose(1, 2, 0))
        break
        
        
def print_class_scale(train_loader, class_map):
    labels_count = {k: v for k, v in zip(range(0, len(class_map)), [0]*len(class_map))}
    for _, labels in train_loader:
        for label in labels:
            labels_count[label.item()] += 1

    labels = list(class_map.keys())
    values = list(labels_count.values())

    fig = plt.figure(figsize = (15, 5))

    # creating the bar plot
    plt.bar(labels, values, width = 0.5)
    plt.legend(labels = ['Samples Per Class'])
    for l in range(len(labels)):
        plt.annotate(values[l], (-0.15 + l, values[l] + 50))
    plt.xticks(rotation='45')
    plt.xlabel("Classes")
    plt.ylabel("Class Count")
    plt.title("Classes Count")
    plt.show()
    

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.type(torch.float32) / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        
    plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
def plot_incorrect_predictions(predictions, class_map):
    print(f'Total Incorrect Predictions {len(predictions)}')

    classes = list(class_map.values())

    fig = plt.figure(figsize = (15, 18))
    for i, (d, t, p, o) in enumerate(predictions):
        ax = fig.add_subplot(8, 8, i + 1, xticks = [], yticks = [])
        ax.set_title(f'{classes[t.item()]}/{classes[p.item()]}')
        plt.imshow(d.cpu().numpy().transpose(1, 2, 0))
        if i+1 == 8*8:
            break
            
            
def plot_network_performance(epochs, schedule, train_loss, valid_loss, train_correct, valid_correct):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(range(epochs), schedule, 'r', label='One Cycle LR')
    plt.title('Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('LR')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(range(epochs), train_loss, 'g', label='Training loss')
    plt.plot(range(epochs), valid_loss, 'b', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(range(epochs), train_correct, 'g', label='Training Accuracy')
    plt.plot(range(epochs), valid_correct, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    
    
def plot_model_comparison(trainers, epochs):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), trainers[0].list_valid_loss, 'g', label='BN + L1 loss')
    plt.plot(range(epochs), trainers[1].list_valid_loss, 'b', label='GN loss')
    plt.plot(range(epochs), trainers[2].list_valid_loss, 'r', label='LN loss')
    plt.title('Validation losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), trainers[0].list_valid_correct, 'g', label='BN + L1 Accuracy')
    plt.plot(range(epochs), trainers[1].list_valid_correct, 'b', label='GN Accuracy')
    plt.plot(range(epochs), trainers[2].list_valid_correct, 'r', label='LN Accuracy')
    plt.title('Validation Accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()