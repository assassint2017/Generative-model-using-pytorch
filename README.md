# Generative-model-GAN-VAE-using-pytorch

implement **GANs** and **VAE** using **pytorch**

GANs include **DCGAN LSGAN WGAN WGAN-GP**
VAE is from the paper **Auto-Encoding Variational Bayes**

in my code, i use MNIST to test the network, but its very easy to switch dataset of you own

here is result when i training DCGAN at 36 epoch

![image](https://github.com/assassint2017/Generative-model-GAN-VAE-using-pytorch/blob/master/img/DCGAN_36.png)

next are some humanface generate by WGAN-GP

the image use to train the WGAN is from baidu image

![face1](https://github.com/assassint2017/Generative-model-GAN-VAE-using-pytorch/blob/master/img/face1.png)
![face2](https://github.com/assassint2017/Generative-model-GAN-VAE-using-pytorch/blob/master/img/face2.png)
![face3](https://github.com/assassint2017/Generative-model-GAN-VAE-using-pytorch/blob/master/img/face3.png)
![face4](https://github.com/assassint2017/Generative-model-GAN-VAE-using-pytorch/blob/master/img/face4.png)
![face5](https://github.com/assassint2017/Generative-model-GAN-VAE-using-pytorch/blob/master/img/face5.png)

finall is the gif from training a VAE use MNIST datasset

leftside is the reconstruction image,middle is the training image, and rightside is the image generate from the noise

![gif](https://github.com/assassint2017/Generative-model-GAN-VAE-using-pytorch/blob/master/img/VAE-MNIST.gif)
