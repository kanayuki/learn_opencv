import os
import re

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchsummary import summary
from torchvision import transforms
import cv2 as cv

to_tensor = transforms.ToTensor()


class MyDataSet(torch.utils.data.Dataset):
    def __init__(self):
        self.dir = '../Image/nums'
        self.items = []

        filenames = os.listdir(self.dir)
        for name in filenames:
            m = re.match(r'\d+-num-(\d)\.png', name)
            if m:
                label = int(m.group(1)) - 1
                self.items.append((name, label))

    def __getitem__(self, item):
        name, label = self.items[item]

        path = os.path.join(self.dir, name)
        img = Image.open(path)
        img = cv.GaussianBlur(np.array(img), (5, 5), 0)
        img = to_tensor(img)
        label = torch.tensor(label)

        return img, label

    def __len__(self):
        return len(self.items)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.module = nn.Sequential(
            # nn.Conv2d(1, 64, 3, padding=1),
            # nn.ReLU(True),
            # nn.MaxPool2d(2, 2),
            #
            # nn.Conv2d(64, 128, 3, padding=1),
            # nn.ReLU(True),
            # nn.MaxPool2d(3, 2),

            # nn.Conv2d(128, 256, 3, padding=1),
            # nn.ReLU(True),
            # nn.MaxPool2d(3, 2),

            nn.Flatten(),
            nn.Linear(30*30, 900),
            # nn.Dropout(),
            nn.ReLU(True),
            # nn.Linear(1000, 1000),
            # # nn.Dropout(),
            # nn.ReLU(True),
            nn.Linear(900, 100),
            # nn.Dropout(),
            # nn.ReLU(True),
            # nn.Linear(500, 100),
            # nn.Dropout(),
            nn.ReLU(True),
            nn.Linear(100, 9),

            nn.Softmax(1)

        )

    def forward(self, x):
        x = self.module(x)
        return x


def train(cnn, num_epochs=10):
    my_dataset = MyDataSet()
    train_loader = torch.utils.data.DataLoader(dataset=my_dataset, batch_size=16, shuffle=True)

    optimizer = torch.optim.Adam(cnn.parameters(), 1e-3)
    loss_fn = nn.CrossEntropyLoss()

    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (img, labels) in enumerate(train_loader):
            img = img.cuda()
            labels = labels.cuda()

            output = cnn(img)
            loss = loss_fn(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch [{}/{}] Step [{}/{}] Loss: {}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    torch.save(cnn.state_dict(), 'sudoku-cnn.ckpt')


def test():
    cnn = CNN().cuda()
    cnn.load_state_dict(torch.load('sudoku-cnn.ckpt'))
    img = Image.open('../Image/nums/219-num-8.png')

    img = to_tensor(img).reshape(1, 1, 30, 30).cuda()
    output = cnn(img)

    print(output)
    _, predicted = output.max(1)

    n = predicted.item() + 1
    print(n)


def recognizer():
    cnn = CNN().cuda()
    cnn.load_state_dict(torch.load('sudoku-cnn.ckpt'))
    size = 30

    def recog(image):
        image = cv.resize(image, (size, size))
        img = to_tensor(image).reshape(1, 1, size, size).cuda()

        output = cnn(img)
        _, predicted = output.max(1)
        n = predicted.item() + 1
        return n

    return recog


if __name__ == '__main__':
    cnn = CNN().cuda()
    print(cnn)
    # summary(cnn, (1, 30, 30))

    train(cnn, 100)
    #
    test()
