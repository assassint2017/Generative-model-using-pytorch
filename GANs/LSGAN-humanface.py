from time import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import cv2 as cv


# Hyper parameters
epoch_num = 200
img_size = 64  # size of generated image
batch_size = 128
lr_g = 0.0002  # learning rate for Generator
lr_d = 0.0002  # learning rate for Discriminator
latent = 10  # dim of latent space
img_channel = 3  # channel of generated image
init_channel = 64  # control the initial Conv channel of the Generator and Discriminator
workers = 1  # subprocess number for load the image
k = 1  # train Discriminator K times and then train Generator one time

mean = [0.5]
std = [0.5]

slope = 0.2  # slope for leaky relu

# data enhancement
data_transform = transforms.Compose([
    transforms.Resize(size=img_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# dataset
dataset = torchvision.datasets.ImageFolder(root='./data/face/', transform=data_transform)

data_loader = DataLoader(dataset, batch_size, True, num_workers=workers)


# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(latent, init_channel * 8, 4, bias=False),
            nn.BatchNorm2d(init_channel * 8),
            nn.ReLU(),

            nn.Conv2d(init_channel * 8, init_channel * 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(init_channel * 8),
            nn.ReLU()
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(init_channel * 8, init_channel * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(init_channel * 4),
            nn.ReLU(),

            nn.Conv2d(init_channel * 4, init_channel * 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(init_channel * 4),
            nn.ReLU()
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(init_channel * 4, init_channel * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(init_channel * 2),
            nn.ReLU(),

            nn.Conv2d(init_channel * 2, init_channel * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(init_channel * 2),
            nn.ReLU()
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(init_channel * 2, init_channel, 4, 2, 1, bias=False),
            nn.BatchNorm2d(init_channel),
            nn.ReLU(),

            nn.Conv2d(init_channel, init_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(init_channel),
            nn.ReLU()
        )

        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(init_channel, img_channel, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        # initialization for parameters

        for layer in self.modules():

            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.normal(layer.weight.data, 0, 0.02)

            elif isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def forward(self, inputs):

        outputs = self.deconv1(inputs)
        outputs = self.deconv2(outputs)
        outputs = self.deconv3(outputs)
        outputs = self.deconv4(outputs)
        outputs = self.deconv5(outputs)

        return outputs


# Discriminator(
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(img_channel, init_channel, 4, 2, 1, bias=False),
            nn.LeakyReLU(slope)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(init_channel, init_channel * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(init_channel * 2),
            nn.LeakyReLU(slope)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(init_channel * 2, init_channel * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(init_channel * 4),
            nn.LeakyReLU(slope)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(init_channel * 4, init_channel * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(init_channel * 8),
            nn.LeakyReLU(slope)
        )

        self.conv5 = nn.Sequential(

            # no sigmoid!
            nn.Conv2d(init_channel * 8, 1, 4, bias=False)
        )

        # initialization for parameters
        for layer in self.modules():

            if isinstance(layer, nn.Conv2d):
                nn.init.normal(layer.weight.data, 0, 0.02)

            elif isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def forward(self, inputs):

        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        outputs = self.conv5(outputs)

        return outputs.view(inputs.size(0))


# use cuda if you have GPU
net_g = Generator().cuda()
net_d = Discriminator().cuda()

# optimizer
opt_g = torch.optim.Adam(net_g.parameters(), lr=lr_g, betas=(0.5, 0.999))  # optimizer for Generator
opt_d = torch.optim.Adam(net_d.parameters(), lr=lr_d, betas=(0.5, 0.999))  # optimizer for Discriminator


# get random noise
def get_noise(noise_num=batch_size):

     return Variable(torch.randn((noise_num, latent, 1, 1)).cuda())


# train the network
start = time()
fix_noise = get_noise(64)
number = 1

for epoch in range(epoch_num):

    for step, (real_data, target) in enumerate(data_loader, 1):

        # train Discriminator

        real_data = Variable(real_data).cuda()

        prob_fake = net_d(net_g(get_noise(real_data.size(0))))

        prob_real = net_d(real_data)

        loss_d = 0.5 * torch.mean(torch.pow(prob_real - 1, 2) + torch.pow(prob_fake, 2))

        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()

        # train Generator
        if step % k is 0:

            prob_fake = net_d(net_g(get_noise()))

            loss_g = 0.5 * torch.mean(torch.pow(prob_fake - 1, 2))

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

        if step % 20 is 0:

            fake_img = torchvision.utils.make_grid((0.5 * net_g(fix_noise).data.cpu() + 0.5))

            cv.imwrite('./img/LS/' + str(number) + '.png', fake_img.permute(1, 2, 0).numpy() * 255)

            number += 1
