import os
import re

import torch
from PIL import Image
from torch import nn
from torchsummary import summary
from torchvision import transforms

to_tensor = transforms.ToTensor()


class MyDataSet(torch.utils.data.Dataset):
    def __init__(self):
        self.dir = '../Image'
        self.items = []

        filenames = os.listdir(self.dir)
        for name in filenames:
            m = re.match(r'\d+-num-(\d+)\.png', name)
            if m:
                label = int(m.group(1)) - 1
                self.items.append((name, label))

    def __getitem__(self, item):
        name, label = self.items[item]

        path = os.path.join(self.dir, name)
        img = Image.open(path)
        # img = img.convert('L')
        img = to_tensor(img)

        label = torch.tensor(label)

        return img, label

    def __len__(self):
        return len(self.items)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.module = nn.Sequential(
            # nn.Conv2d(1, 6, 3, padding=1),
            # nn.ReLU(True),
            # nn.MaxPool2d(3, 2),
            # nn.Flatten(),
            nn.Linear(38 * 45, 500),
            nn.ReLU(True),
            nn.Linear(500, 9),

        )

    def forward(self, x):
        x = self.module(x)
        return x


def train(cnn):
    my_dataset = MyDataSet()
    train_loader = torch.utils.data.DataLoader(dataset=my_dataset, batch_size=10, shuffle=True)

    num_epochs = 5

    optimizer = torch.optim.Adam(cnn.parameters(), 1e-3)
    loss_fn = nn.CrossEntropyLoss()

    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (img, labels) in enumerate(train_loader):
            img = img.reshape(-1, 38 * 45).cuda()
            labels = labels.cuda()

            output = cnn(img)
            loss = loss_fn(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch [{}/{}] Step [{}/{}] Loss: {}'.format(epoch, num_epochs, i, total_step, loss.item()))

    torch.save(cnn, 'sudoku-cnn.ckpt')


def test():
    cnn = torch.load('sudoku-cnn.ckpt')
    img = Image.open('../Image/22-num-5.png')
    img = to_tensor(img).reshape(1, 38 * 45).cuda()

    output = cnn(img)

    print(output)
    _, predicted = output.max(1)

    print(predicted.item())


if __name__ == '__main__':
    # cnn = CNN().cuda()
    # print(cnn)
    # summary(cnn, (1, 45 * 38))

    # train(cnn)

    test()
