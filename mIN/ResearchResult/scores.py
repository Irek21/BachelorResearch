import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def AccuracyScore(classifier, loader):
    train_size = loader.train_size
    batch_size = loader.batch_size
    
    correct = 0
    total = 0
    for i in range(train_size // batch_size):
        images, labels = loader.batch_load(i)

        predict = classifier(images.to(device))
        _, predicted = torch.max(predict.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    accuracy = 100 * correct / total
    return accuracy