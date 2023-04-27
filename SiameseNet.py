import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import LFWPairs
from torchvision.transforms import transforms


class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv layer will simply collapse the image across RGB channels (becomes 1 channel) by getting an average pixel value
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(1, 1), bias=False)
        self.conv1.weight.requires_grad = False
        self.conv1.weight.data = torch.tensor(
            [
                [
                    [
                        [1 / 3]
                    ],
                    [
                        [1 / 3]
                    ],
                    [
                        [1 / 3]
                    ],
                ]
            ], dtype=torch.float32
        )
        self.fc1 = nn.Linear(62500, 12288)
        self.fc2 = nn.Linear(12288, 4096)
        self.fc3 = nn.Linear(4096, 1024)
        self.fc4 = nn.Linear(1024, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 62500)
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class DenseLayers(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


lfw_train = LFWPairs(download=True, root='./', split='train', transform=transforms.ToTensor())
lfw_test = LFWPairs(download=True, root='./', split='test', transform=transforms.ToTensor())

# Training data set has 2200 pairs of images, so ensure batch size can divide this without remainder
BATCH_SIZE = 100
lfw_train_loader = DataLoader(dataset=lfw_train, shuffle=True, batch_size=BATCH_SIZE)
lfw_test_loader = DataLoader(dataset=lfw_test)

train_loader_length = len(lfw_train_loader)
test_loader_length = len(lfw_test_loader)

siamese_net = SiameseNet()
dense_layers = DenseLayers()

siamese_optimizer = optim.Adam(siamese_net.parameters(), lr=0.0001)
dense_optimizer = optim.Adam(dense_layers.parameters(), lr=0.0001)

EPOCHS = 50

# Creates a new line after file download output
print('')

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}:')
    # batch, first face, second face, matching
    for i, (first, second, answer) in enumerate(lfw_train_loader):
        siamese_optimizer.zero_grad()
        dense_optimizer.zero_grad()

        siamese_output_first = siamese_net(first)
        siamese_output_second = siamese_net(second)

        difference = siamese_output_first - siamese_output_second
        abs_difference = torch.abs(difference)

        dense_output = dense_layers(abs_difference)

        one_hot = torch.zeros([BATCH_SIZE, 1], dtype=torch.float32)
        for index in range(len(one_hot)):
            one_hot[index][0] = answer[index]

        loss = functional.mse_loss(dense_output, one_hot)
        loss.backward()
        siamese_optimizer.step()
        dense_optimizer.step()

        # Will not appear in PyCharm run tab, use terminal
        print(f'{(i + 1) / train_loader_length * 100:.1F}% of training completed.', end='\r')
    print('')

    correct = 0
    total = 0

    with torch.no_grad():
        for i, (first, second, answer) in enumerate(lfw_test_loader):
            siamese_output_first = siamese_net(first)
            siamese_output_second = siamese_net(second)

            difference = siamese_output_first - siamese_output_second
            abs_difference = torch.abs(difference)

            dense_output = dense_layers(abs_difference)

            if answer == 1:
                if dense_output >= 0.5:
                    correct += 1
            else:
                if dense_output < 0.5:
                    correct += 1
            total += 1

            # Will not appear in PyCharm run tab, use terminal
            print(f'{(i + 1) / test_loader_length * 100:.1F}% of testing completed.', end='\r')
        print('')

    print(f'Accuracy: {correct / total * 100}%\n')
