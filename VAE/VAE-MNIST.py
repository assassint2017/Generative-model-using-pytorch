from time import time

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt


# Hyper parameters
epoch_num = 200
img_size = 64  # size of generated image
batch_size = 128
lr_en = 0.0002  # learning rate for encoder
lr_de = 0.0002  # learning rate for decoder
latent = 100  # dim of latent space
img_channel = 1  # channel of generated image
init_channel = 16  # control the initial Conv channel of the Generator and Discriminator
workers = 1  # subprocess number for load the image
dataset_size = 60000  # image number of your training set

mean = [0.5]
std = [0.5]

slope = 0.2  # slope for leaky relu

# data enhancement
data_transform = transforms.Compose([
    transforms.Resize(size=img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# dataset
data_set = torchvision.datasets.MNIST(
    root='./MNIST',
    train=True,
    transform=data_transform
)

data_loader = DataLoader(data_set, batch_size, True, num_workers=workers)


# Encoder
class Encoder(nn.Module):
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

        self.conv5 = nn.Conv2d(init_channel * 8, init_channel * 8, 4, bias=False)

        self.mean = nn.Linear(init_channel * 8, latent)

        self.log_var = nn.Linear(init_channel * 8, latent)

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

        outputs = outputs.view(inputs.size(0), -1)

        return self.mean(outputs), self.log_var(outputs)


# Decoder
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(latent, init_channel * 8, 4, bias=False),
            nn.BatchNorm2d(init_channel * 8),
            nn.ReLU(),
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(init_channel * 8, init_channel * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(init_channel * 4),
            nn.ReLU(),
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(init_channel * 4, init_channel * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(init_channel * 2),
            nn.ReLU(),
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(init_channel * 2, init_channel, 4, 2, 1, bias=False),
            nn.BatchNorm2d(init_channel),
            nn.ReLU(),
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

        inputs = inputs.view(inputs.size(0), inputs.size(1), 1, 1)

        outputs = self.deconv1(inputs)
        outputs = self.deconv2(outputs)
        outputs = self.deconv3(outputs)
        outputs = self.deconv4(outputs)
        outputs = self.deconv5(outputs)

        return outputs


# use cuda if you have GPU
decoder = Decoder().cuda()
encoder = Encoder().cuda()

# optimizer
opt_d = torch.optim.Adam(decoder.parameters(), lr=lr_de)  # optimizer for decoder
opt_e = torch.optim.Adam(encoder.parameters(), lr=lr_en)  # optimizer for encoder


# get random noise
def get_noise(noise_num=batch_size):

     return Variable(torch.randn((noise_num, latent)).cuda())


# reconstruction loss
re_loss = nn.MSELoss()

# train the network
start = time()
number = 1
img = plt.figure('Visualization')

for epoch in range(epoch_num):

    for step, (real_data, target) in enumerate(data_loader, 1):

        opt_e.zero_grad()
        opt_d.zero_grad()

        real_data = Variable(real_data).cuda()

        mean, log_var = encoder(real_data)

        fake_noise = mean + (torch.exp(log_var / 2) * get_noise(real_data.size(0)))

        reconstruct_img = decoder(fake_noise)

        KL_loss = torch.sum(- 0.5 * (1 + log_var ** 2 - mean ** 2 - torch.exp(log_var)), dim=1)

        loss = re_loss(reconstruct_img, real_data) + torch.mean(KL_loss)

        loss.backward()

        opt_d.step()
        opt_e.step()

        if step % 20 is 0:

            iteration = epoch * (math.ceil(dataset_size / batch_size)) + step
            print('epoch:', epoch, 'step', step, 'time:', (time() - start) / 60, 'min')

            img.add_subplot(131)
            plt.title('reconstruction_img')

            re_img = torchvision.utils.make_grid(reconstruct_img.data[0:36].cpu() * 0.5 + 0.5, nrow=6)
            plt.imshow(re_img.permute(1, 2, 0).numpy())

            img.add_subplot(132)
            plt.title('train_img')

            real_img = torchvision.utils.make_grid(real_data.data[0:36].cpu() * 0.5 + 0.5, nrow=6)
            plt.imshow(real_img.permute(1, 2, 0).numpy())

            img.add_subplot(133)
            plt.title('generate_img')

            ge_img = torchvision.utils.make_grid(decoder(get_noise(36)).data.cpu() * 0.5 + 0.5, nrow=6)
            plt.imshow(ge_img.permute(1, 2, 0).numpy())

            img.savefig('./img/' + str(number) + '.png')

            number += 1

            plt.pause(0.01)

