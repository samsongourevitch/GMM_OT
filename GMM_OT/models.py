import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE_Encoder_MNIST(nn.Module):
    def __init__(self, n_latent):
        super(VAE_Encoder_MNIST, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(128*3*3, n_latent)
        self.fc_logvar = nn.Linear(128*3*3, n_latent)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
class VAE_Decoder_MNIST(nn.Module):
    def __init__(self, n_latent):
        super(VAE_Decoder_MNIST, self).__init__()

        self.fc = nn.Linear(n_latent, 128*3*3)
        self.unflatten = nn.Unflatten(1, (128, 3, 3))

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = self.unflatten(x)
        x = self.deconv_layers(x)
        return x
    
class VAE_MNIST(nn.Module):
    def __init__(self, n_latent=30):
        super(VAE_MNIST, self).__init__()

        self.n_latent = n_latent

        self.encoder = VAE_Encoder_MNIST(n_latent)
        self.decoder = VAE_Decoder_MNIST(n_latent)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = mu + torch.exp(0.5*logvar)*torch.randn_like(mu)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def sample(self, n_samples):
        z = torch.randn(n_samples, self.n_latent, device=device)
        x_recon = self.decoder(z)
        return x_recon
    
class AE_Encoder_MNIST(nn.Module):
    def __init__(self, n_latent):
        super(AE_Encoder_MNIST, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128*3*3, n_latent)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        z = self.fc(x)
        return z
    
class AE_Decoder_MNIST(nn.Module):
    def __init__(self, n_latent):
        super(AE_Decoder_MNIST, self).__init__()

        self.fc = nn.Linear(n_latent, 128*3*3)
        self.unflatten = nn.Unflatten(1, (128, 3, 3))

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = self.unflatten(x)
        x = self.deconv_layers(x)
        return x
    
class Autoencoder_MNIST(nn.Module):
    def __init__(self, n_latent=30):
        super(Autoencoder_MNIST, self).__init__()

        self.n_latent = n_latent

        self.encoder = AE_Encoder_MNIST(n_latent)
        self.decoder = AE_Decoder_MNIST(n_latent)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

    def sample(self, n_samples):
        z = torch.randn(n_samples, self.n_latent, device=device)
        x_recon = self.decoder(z)
        return x_recon