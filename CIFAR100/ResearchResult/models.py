import numpy as np
from PIL import Image as img
import torch
import torch.nn as nn
import torchvision.transforms as transforms

class DeltaEncoder(nn.Module):
    def __init__(self, input_size=2048, hidden_size=8192, neck_size=16):
        encoder = nn.Sequential(
            nn.Linear(input_size * 2, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.Linear(hidden_size, neck_size),
        )

        decoder = nn.Sequential(
            nn.Linear(input_size + neck_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.Linear(hidden_size, input_size),
        )
        dropout = nn.Dropout(0.5)

        super(DeltaEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dropout = dropout

    def forward(self, X1, X2):
        out = self.dropout(X1)
        out = torch.cat((out, X2), dim=1)
        out = self.encoder(out)

        out = torch.cat((X2, out), dim=1)
        out = self.decoder(out)
        return out

class DeltaEncoderGenerator(nn.Module):
    def __init__(self, input_size=2048, hidden_size=8192, neck_size=16):
        encoder = nn.Sequential(
            nn.Linear(input_size * 2, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.Linear(hidden_size, neck_size),
        )

        decoder = nn.Sequential(
            nn.Linear(input_size + neck_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.Linear(hidden_size, input_size),
        )
        dropout = nn.Dropout(0.5)

        super(DeltaEncoderGenerator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dropout = dropout

    def forward(self, X1, X2, shot):
        out = self.dropout(X1)
        out = torch.cat((out, X2), dim=1)
        out = self.encoder(out)

        out = torch.cat((shot, out), dim=1)
        out = self.decoder(out)
        return out

class CentroidGenerator(nn.Module):
    def __init__(self, input_size=2048, hidden_size=2048):
        fc_layers = nn.Sequential(
            nn.Linear(input_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.ReLU()
        )

        super(CentroidGenerator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc = fc_layers

    def forward(self, c1_a, c2_a, c1_b):
        x = torch.cat((c1_a, c2_a, c1_b), dim=1)
        c2_b_predict = self.fc(x)
        return c2_b_predict
    
class Classifier(nn.Module):
    def __init__(self):
        fc_layers = nn.Sequential(
            nn.Linear(2048, 5),
            nn.Softmax(dim=1)
        )
        super(Classifier, self).__init__()
        self.fc = fc_layers
        
    def forward(self, x):
        out = self.fc(x)
        return out

class AugGenerator():
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.idx = np.random.permutation(num_samples) % 6
        self.rotate = transforms.RandomRotation(30)
        self.flip = transforms.RandomHorizontalFlip(1)
        self.noise = transforms.GaussianBlur(3)
        self.perspective = transforms.RandomPerspective(p=1)
        self.affine = transforms.RandomAffine(20, (0.2, 0.2))
        self.jitter = transforms.ColorJitter((0.8, 1), (0.8, 1), (0.8, 1))
        self.resize = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.totensor = transforms.ToTensor()
        
    def reshuffle(self):
        self.idx = np.random.permutation(self.num_samples) % 6
        
    def aug(self, np_im, i):
        image = img.fromarray(np.transpose(np_im, axes=(1, 2, 0)))
        image = self.resize(image)
        
        if self.idx[i] == 0:
            aug = self.rotate(image)
        elif self.idx[i] == 1:
            aug = self.flip(image)
        elif self.idx[i] == 2:
            aug = self.noise(image)
        elif self.idx[i] == 3:
            aug = self.perspective(image)
        elif self.idx[i] == 4:
            aug = self.affine(image)
        elif self.idx[i] == 5:
            aug = self.jitter(image)
            
        # im_tensor = self.totensor(np.transpose(im.numpy(), axes=(1, 2, 0)))
        return self.normalize(self.totensor(image))