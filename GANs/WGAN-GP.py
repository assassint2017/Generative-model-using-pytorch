from time import time

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import tensorboardX

# Hyper parameters
epoch_num = 200
img_size = 64  # size of generated image
batch_size = 128
lr_g = 0.0002  # learning rate for Generator
lr_c = 0.0002  # learning rate for Critic
latent = 100  # dim of latent space
img_channel = 1  # channel of generated image
init_channel = 16  # control the initial Conv channel of the Generator and Discriminator
workers = 2  # subprocess number for load the image
k = 5  # train Critic K times and then train Generator one time
dataset_size = 60000  # image number of your training set

mean = [0.5]
std = [0.5]

slope = 0.2  # slope for leaky relu

penalty_coeffcient = 10  # penalty coeffcient

beta = (0, 0.9)  # Adam hyperpaarmeters

# use tensorboard
writer = tensorboardX.SummaryWriter(log_dir='./logs/')

# data enhancement
data_transform = transforms.Compose([
    transforms.Resize(size=img_size),
    transforms.RandomRotation(5),
    transforms.RandomHorizontalFlip(),
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


# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(latent, init_channel * 8, 4, bias=False),
            nn.BatchNorm2d(init_channel * 8),
            nn.ReLU()
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(init_channel * 8, init_channel * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(init_channel * 4),
            nn.ReLU()
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(init_channel * 4, init_channel * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(init_channel * 2),
            nn.ReLU()
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(init_channel * 2, init_channel, 4, 2, 1, bias=False),
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


# Critic : no BN!!!
class Critic(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(img_channel, init_channel, 4, 2, 1, bias=False),
            nn.LeakyReLU(slope)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(init_channel, init_channel * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(slope)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(init_channel * 2, init_channel * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(slope)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(init_channel * 4, init_channel * 8, 4, 2, 1, bias=False),
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

    def forward(self, inputs):

        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        outputs = self.conv5(outputs)

        return outputs.view(inputs.size(0))


# use cuda if you have GPU
net_g = Generator().cuda()
net_c = Critic().cuda()


# use tensorboard draw the computational graph
writer.add_graph(net_g, Variable(torch.randn(batch_size, latent, 1, 1).cuda()))

writer.add_graph(net_c, Variable(torch.FloatTensor(batch_size, img_channel, img_size, img_size).cuda()))

# optimizer
opt_g = torch.optim.Adam(net_g.parameters(), lr=lr_g, betas=beta)  # optimizer for Generator
opt_c = torch.optim.Adam(net_c.parameters(), lr=lr_c, betas=beta)  # optimizer for Critic


# get random noise
def get_noise(noise_num=batch_size):

     return Variable(torch.randn((noise_num, latent, 1, 1)).cuda())


# train the network
start = time()
fix_noise = get_noise(64)

for epoch in range(epoch_num):

    for step, (real_data, target) in enumerate(data_loader, 1):

        # train Critic

        real_data = Variable(real_data).cuda()

        fake_data = net_g(get_noise(real_data.size(0)))

        output_fake = net_c(fake_data)

        output_real = net_c(real_data)

        # Calculate gradient penalty

        e = Variable(torch.rand(real_data.size(0), 1, 1, 1).cuda())

        #print(type(real_data), real_data.size())
        #print(type(fake_data), fake_data.size())
        #print(type(e), e.size())

        interpolation = e * real_data + (1 - e) * fake_data

        gradient_penalty = torch.autograd.grad(outputs=net_c(interpolation), inputs=interpolation,
                              grad_outputs=torch.ones(net_c(interpolation).size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = torch.mean((gradient_penalty.norm(2, dim=1) - 1) ** 2)

        opt_c.zero_grad()

        gradient_penalty.backward(retain_graph=True)

        #print(type(gradient_penalty), gradient_penalty.size())

        loss_c = torch.mean(output_fake - output_real) + gradient_penalty * penalty_coeffcient

        loss_c.backward()
        opt_c.step()

        # train Generator
        if step % k is 0:

            output_fake = net_c(net_g(get_noise()))

            loss_g = - torch.mean(output_fake)

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

        if step % 20 is 0:

            iteration = epoch * (math.ceil(dataset_size / batch_size)) + step
            writer.add_scalars('loss', {'g_loss': loss_g.data[0], 'c_loss': loss_c.data[0]}, iteration)
            print('epoch:', epoch, 'step', step, 'time:', (time() - start) / 60, 'min')

            fake_img = torchvision.utils.make_grid((0.5 * net_g(fix_noise).data.cpu() + 0.5))
            plt.imshow(fake_img.permute(1, 2, 0).numpy())
            plt.pause(0.01)

plt.show()

