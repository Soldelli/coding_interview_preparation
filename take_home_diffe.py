#!/usr/bin/env python
# coding: utf-8

# # Task instructions (please read carefully)
# 
# - Allocate up to 4 hours to work on these tasks.
# - To solve the task, first make a copy of this notebook by clicking in the top-left "File" -> “Save a copy in Drive”. Then solve the task on a standard GPU instance in colab.research.google.com.
# - To setup the GPU instance runtime, in top bar click "runtime" -> "change runtime type" -> "T4 GPU" (under Hardware accelerator). Note that this instance is free, i.e. no money will be billed to your Google account.
# - Use any tools available to you, including existing libraries, LLMs, coding assistants, etc. To install a library in colab, you can use `!pip install <library_name>` in the top cell.
# - Some of the tasks are very hard, and it is not expected for all candidates to solve them.
# - For all optimization processes (training, SDS optimization, ...), make sure to log the optimized objectives and display graphs displaying their convergence.
# - Besides the correctness of the solutions, we will also assess the code quality.
# 
# ### Submission:
# - Once done with solving the tasks -> click “Share” in top-right -> under “General access” select “Anyone with the link” -> click “Copy link” -> paste the link to the submission form.
# - Together with sharing the link, download the .ipynb file and upload the file into the submission form.
# - Before time runs out, please write a paragraph about the things you tried, what worked and what didn't, how did you debug it, and what would you do if you had more time. Please write this as a text cell directly inside the notebook.
# - Before submitting, make sure that your colab runs from beginning to end on a standard GPU instance in [colab.research.google.com](colab.research.google.com).
# 
# Good luck!

# # Machine Learning Take-Home Exercise: Diffusion Models and Inpainting
# 
# In this exercise, you will:
# 1. Implement a simple diffusion model and train it on the MNIST dataset
# 2. Use the trained model to perform image inpainting
# 3. Use the trained model to perform image inpainting using SDS optimization
# 
# ## Part 1: Implementing a Simple Diffusion Model
# 
# First, let's import the necessary libraries and set up our environment.

# In[2]:


import os
import torch
import random
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import typing
import torchvision
from tqdm.auto import tqdm

if not torch.cuda.is_available():
    raise RuntimeError("No GPU found. Make sure to select a GPU runtime in the Colab instance settings.")

device = torch.device('cuda')

# ### 1.1 Data Loading
# 
# We'll use MINST dataset. We setup the data loader to load small (16x16) images, so that we can train a small diffusion model in this dataset quickly.

# In[3]:


NUM_CLASSES = 10  # the number of digit classes in the MNIST dataset

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((16, 16)),
    transforms.Normalize((0.5,), (0.5,)),
])
trainset = torchvision.datasets.MNIST(
    root='./data_take_home_minst', train=True,
    download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=4
)
valset = torchvision.datasets.MNIST(
    root='./data_take_home_minst', train=False,
    download=True, transform=transform
)


# ### 1.2 Implementing the Model
# 
# Complete the Model implementation below. This will be our noise prediction network.
# 
# NOTE:
# - In the following cells, feel free to add any additional parameters you need into the function signatures.
# - A succesful diffusion model can be trained within 15 minutes with a model containing ~30M parameters.

# In[ ]:


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: the input image, a float tensor of shape (batch_size, in_channels, H, W)
            t: the timestep, a long tensor of shape (batch_size,)
            y: the class label, a long tensor of shape (batch_size,)
        """
        raise NotImplementedError("Finish me!")


# ### 1.3 Training Loop
# 
# First, implement the forward diffusion process and the loss computation.
# 
# Then, implement the training loop for the diffusion model.
# 
# Once done, train the model to convergence in order to be able to sample images of digits from the MNIST dataset.

# In[ ]:


def diffusion_forward(x_0: torch.Tensor, t: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward diffusion process, given a clean image x_0 and a timestep t,
    returns the noised image and the noise.
    """
    raise NotImplementedError("Finish me!")

def compute_loss(model: nn.Module, x_0: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the diffusion loss that supervises the denoiser `model`.

    Args:
        model: the noise prediction model
        x_0: the clean image, a float tensor of shape (batch_size, in_channels, H, W)
        y: the class label, a long tensor of shape (batch_size,)
    """
    raise NotImplementedError("Finish me!")

    loss = ...

    return loss

def train_diffusion(model, dataloader, num_epochs=7):
    """Training loop for the diffusion model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in progress_bar:

            # get the clean image and the label
            x_0 = batch[0].to(device)
            y = batch[1].to(device)

            # calculate loss
            loss = compute_loss(model, x_0, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            raise NotImplementedError("Finish me!")

model = Model().to(device)
train_diffusion(model, trainloader)


# ### 1.4: Visualizing the diffusion samples
# 
# Given the pretrained model, sample several images of digits. Visualize 4 samples for each digit class in the MNIST (i.e. classes 0-to-9, inclusive), and show the result in a 10x4 grid of images.

# In[ ]:


def sample_from_model(model: nn.Module, num_samples: int, digit_class: int) -> torch.Tensor:
    """
    Sample `num_samples` images of a specific digit class from the model.

    Args:
        model: the diffusion model
        num_samples: the number of samples to draw
        digit_class: the digit class to sample from
    """
    raise NotImplementedError("Finish me!")

plt.figure(figsize=(10, 3))
for digit_class in range(NUM_CLASSES):
    samples = sample_from_model(model, digit_class=digit_class, num_samples=4)
    samples = samples.detach().cpu() * 0.5 + 0.5
    samples_im = torch.cat([s[0] for s in samples], dim=0)
    subplot = plt.subplot(1, 10, digit_class + 1)
    subplot.imshow(samples_im)
    subplot.axis('off')
    subplot.set_title(f'Digit {digit_class}')
plt.show()


# ## Part 2: Implementing Inpainting with Diffusion
# 
# Now that we have a trained diffusion model, let's implement inpainting using the model.
# More specifically, given an image of a digit, we first mask a part of it. We then ask you to leverage the MNIST-trained diffusion model to inpaint the missing part so that the resulting image still looks like an image of the corresponding digit (the digit class is known as well).
# 
# You can select a method of your choice as long as you utilize the diffusion model you just trained.

# In[ ]:


def create_mask(image_size, box_size, position):
    """Create a binary mask for inpainting"""
    mask = torch.zeros((1, 1, image_size, image_size))
    x, y = position
    mask[:, :, y:y+box_size, x:x+box_size] = 1
    return mask

def inpaint_with_diffusion_model(
    model: nn.Module,
    image_masked: torch.Tensor,
    mask: torch.Tensor,
    digit_class: int,
) -> torch.Tensor:
    """
    Implement inpainting using the diffusion model

    Args:
        model: the diffusion model
        image_masked: the image to inpaint
        mask: binary mask where 1s are the masked region, and 0s are the known region
        digit_class: the digit class of the image
    """

    raise NotImplementedError("Finish me!")

    image_inpainted = ...

    return image_inpainted

# take a random image from the validation set
image, label = valset[124]

# create a square mask
mask = create_mask(image.shape[1], 10, (5, 5))

# mask the image
image_masked = image * (1 - mask) + mask * (-1)

# inpaint the image
image_inpainted = inpaint_with_diffusion_model(
    model,
    image_masked,
    mask,
    label,
)

# show the known inputs and the inpainted image
plt.figure(figsize=(6, 3))
plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(image.cpu()[0].clip(-1, 1) * 0.5 + 0.5)
plt.subplot(2, 2, 2)
plt.title("Mask")
plt.imshow(mask.cpu()[0, 0].clip(-1, 1) * 0.5 + 0.5)
plt.subplot(2, 2, 3)
plt.title("Masked Image")
plt.imshow(image_masked.cpu()[0, 0].clip(-1, 1) * 0.5 + 0.5)
plt.subplot(2, 2, 4)
plt.title("Inpainted Image")
plt.imshow(image_inpainted.cpu()[0, 0].clip(-1, 1) * 0.5 + 0.5)
plt.show()


# # Part 3: Inpainting with SDS
# 
# The final task is the same as previous one. However, here we ask you to inpaint the image using a specific method: Score Distillation Sampling (SDS), as described in the DreamFusion paper available on arXiv: https://arxiv.org/abs/2209.14988
# 
# More specifically, we want to optimize a 2D image using SDS so that:
#   - It's known (unmasked) part matches the given ground-truth image
#   - It's unknown part gets feasibly inpainted so that the final image depicts the prescribed digit
# 
# IMPORTANT NOTE: Do NOT optimize a Neural Radiance Field (NeRF) as described in the original DreamFusion paper, but rather a sigle 2D image represented as a `H x W` tensor. The task is to use the SDS method to implement the optimization loop, not to reproduce the DreamFusion paper in its entirety.

# In[ ]:


def sds_loss(
    model: nn.Module,
    opt_image: torch.Tensor,
    gt_image: torch.Tensor,
    mask: torch.Tensor,
    digit_class: int,
    num_optimization_steps: int = 1000,
):

    raise NotImplementedError("Finish me!")

    # Average the SDS loss across all noise levels
    loss = ...

    return loss

def inpaint_with_sds(
    model: nn.Module,
    opt_image: torch.Tensor,
    gt_image: torch.Tensor,
    mask: torch.Tensor,
    digit_class: int,
    num_optimization_steps: int = 1000,
):
    # Reverse diffusion process with classifier-free guidance
    opt_image = torch.zeros_like(image_masked, device=device)
    gt_image = gt_image.to(device)
    mask = mask.to(device)
    opt_image.requires_grad = True
    optimizer = torch.optim.Adam([opt_image], lr=1e-2)
    for step in range(num_optimization_steps):
        optimizer.zero_grad()
        loss = sds_loss(model, opt_image, gt_image, mask, digit_class)
        if step % 100 == 0:
            print(f"Step {step+1}/{num_optimization_steps}, Loss: {loss.item()}")
        loss.backward()
        optimizer.step()
    return opt_image

optimized_image = inpaint_with_sds(model, image_masked, image, mask, label)

# show the known inputs and the optimized image
plt.figure(figsize=(6, 3))
plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(image.cpu()[0].clip(-1, 1) * 0.5 + 0.5)
plt.subplot(2, 2, 2)
plt.title("Mask")
plt.imshow(mask.cpu()[0, 0].clip(-1, 1) * 0.5 + 0.5)
plt.subplot(2, 2, 3)
plt.title("Masked Image")
plt.imshow(image_masked.cpu()[0, 0].clip(-1, 1) * 0.5 + 0.5)
plt.subplot(2, 2, 4)
plt.title("Optimized Image")
plt.imshow(optimized_image.cpu()[0, 0].clip(-1, 1) * 0.5 + 0.5)
plt.show()

