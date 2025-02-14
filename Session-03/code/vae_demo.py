# pip install torch torchvision matplotlib
# Use google colab if file does not run

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils import save_image
import matplotlib.pyplot as plt
import os

os.makedirs("vae_output", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
epochs = 20
learning_rate = 1e-3
latent_dim = 2
img_size = 28
channels = 1

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "mnist_data",
        train=True,
        download=True,
        transform=transform
    ),
    batch_size=batch_size,
    shuffle=True
)

class VAE(nn.Module):
    def __init__(self):
        super(VAE,self).__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_size*img_size, 400),
            nn.ReLU()
        )

        self.mu = nn.Linear(400, latent_dim)
        self.logvar = nn.Linear(400, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, img_size*img_size),
            nn.Tanh()
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        x = self.decoder(z)
        return x.view(-1, channels, img_size, img_size)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
        
def loss_function(reconstructed, original, mu, logvar):
    recon_loss = nn.functional.mse_loss(reconstructed, original, reduction="sum")
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div


vae = VAE().to(device)
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)


for epoch in range(epochs):
    vae.train()
    train_loss = 0
    for imgs, _ in dataloader:
        imgs = imgs.to(device)

        optimizer.zero_grad()
        reconstructed, mu, logvar = vae(imgs)
        loss = loss_function(reconstructed, imgs, mu, logvar)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(dataloader.dataset):.4f}")

    # Save reconstructed images
    with torch.no_grad():
        vae.eval()
        z = torch.randn(64, latent_dim).to(device)
        generated_imgs = vae.decode(z)
        save_image(generated_imgs, f"vae_output/generated_{epoch+1}.png", nrow=8, normalize=True)

print("Training complete! Check the 'vae_output' directory for the generated images.")
