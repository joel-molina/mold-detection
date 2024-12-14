import torch.nn as nn
#conversion from the mold_detection NB to a python script usable in main.py

#define binary classification CNN
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__() #initialize nn.module
        self.pool = nn.MaxPool2d(2,2)
        
        self.relu = nn.ReLU()

        #in: 256x256x3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        #out: 32x32x64

        self.fc1 = nn.Linear(in_features=(8*8*256), out_features=64)
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(in_features=64, out_features=2)
    
    def forward(self, x):
        output = self.conv1(x)
        output = self.relu(output)
        output = self.pool(output)

        output = self.conv2(output)
        output = self.relu(output)
        output = self.pool(output)

        output = self.conv3(output)
        output = self.relu(output)
        output = self.pool(output)

        output = self.conv4(output)
        output = self.relu(output)
        output = self.pool(output)

        output = self.conv5(output)
        output = self.relu(output)
        output = self.pool(output)
        
        output = output.view(-1, 8*8*256) #flatten batch_size, channel*height*width to batch_size, input features
        output = self.fc1(output)
        output = self.drop(output)
        output = self.fc2(output)
        return output