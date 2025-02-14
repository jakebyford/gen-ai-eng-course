# Env Creation
#   python -m venv myenv
#   source myvenv/bin/activate
#   Windows - activate myvenv\Scripts\activate
#
# Packages
# pip install torch torchvision matplotlib numpy
#

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import time

os.makedirs("ddpm_output", exist_ok=True)

# Check to see if nvidia gpu environmanet or not. (ex. Apple)
# torch.device()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters (can change these sizes)
image_size = 28
batch_size = 64
timesteps = 1000
epochs = 20 # epoch = iteration of model training (eg. 20 iterations)
learning_rate = 1e-4 # learning rate

# Download MNIST dataset from : https://yann.lecun.com/exdb/mnist
# Dataset and Data Loader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "mnist",
        train=True,
        download=True,
        transform=transform
    ),
    batch_size=batch_size,
    shuffle=True
)


class DiffusionModel(nn.Module):
    def __init__(self):
        super(DiffusionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        )
    
    def forward(self, x, t):
        return self.net(x)
    
def forward_diffusion(x, t, betas):
    noise = torch.randn_like(x).to(device)
    alpha_t = torch.cumprod(1-betas,dim=0)[t].view(-1,1,1,1)
    x_t = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise
    return x_t, noise

def beta_schedule(timesteps):
    return torch.linspace(1e-4, 0.02, timesteps).to(device)



# Training the Diffusion Model Steps
diffusion_model = DiffusionModel().to(device)
optimizer = optim.Adam(diffusion_model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

betas = beta_schedule(timesteps)

for epoch in range(epochs):
    diffusion_model.train()
    train_loss = 0
    start_time = time.time()

    for batch_idx, (imgs, _) in enumerate(dataloader):
        imgs = imgs.to(device)
        optimizer.zero_grad()

        ## Random timestep
        t = torch.randint(0, timesteps, (imgs.size(0),), device = device).long()

        # Forward diffusion
        x_t, noise = forward_diffusion(imgs, t, betas)

        # predict noise
        noise_pred = diffusion_model(x_t, t.view(-1,1,1,1))
        loss = criterion(noise_pred, noise)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")

    end_time = time.time()
    print(f"Epoch [{epoch+1}/{epochs}]) completed in {end_time - start_time:.2f}s with Loss: {train_loss / len(dataloader)}")

    # Generate Images at the end of each epoch
    with torch.no_grad():
        diffusion_model.eval()
        sample = torch.randn((batch_size, 1, image_size, image_size), device=device)
        print(f"Generating images for epoch {epoch+1}")

        for t in reversed(range(timesteps)):
            alpha_t = torch.cumprod(1-betas,dim=0)[t]
            noise_pred = diffusion_model(sample, t.view(-1,1,1,1))
            sample = (sample - torch.sqrt(1-alpha_t) * noise_pred) / torch.sqrt(alpha_t)

        save_image(sample, f"ddpm/output/generated_{epoch+1}.png", nrow=8, Normalize=True)

print("Training Complete")
