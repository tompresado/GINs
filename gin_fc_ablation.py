# gin_fc_ablation.py
# A script to conduct an ablation study on a fully-connected Generative Inference Network (GIN)
# for occlusion completion on the MNIST dataset, comparing it to a baseline autoencoder.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
import random
import os
import time

# For metrics
# Note: These were installed in the previous step
try:
    import piq
    from torchmetrics.classification import MulticlassAccuracy
except ImportError:
    print("Please install piq and torchmetrics: pip install piq torchmetrics")
    exit()


@dataclass
class Config:
    # --- Experiment ---
    model_type: str = "gin"  # 'gin' or 'autoencoder'
    experiment_name: str = "gin_t8"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Data ---
    image_size: int = 28 * 28
    occlusion_size: int = 10 # Size of the occluded square

    # --- Model ---
    layer_dims: tuple = (image_size, 256, 128) # For both GIN and AE

    # --- GIN Specific ---
    inference_steps: int = 8 # Ablation parameter T
    gin_lr_inference: float = 0.1 # Learning rate for inference updates

    # --- Training ---
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 1e-4

    # --- Visualization ---
    num_vis_samples: int = 10


class OcclusionTransform:
    """A transform to apply a random square occlusion to an image."""
    def __init__(self, occlusion_size=10):
        self.occlusion_size = occlusion_size

    def __call__(self, img):
        # img is a tensor of shape (C, H, W)
        c, h, w = img.shape
        # Make a copy to not modify the original image (which is the target)
        occluded_img = img.clone()

        # Randomly determine the top-left corner of the occlusion patch
        top = random.randint(0, h - self.occlusion_size)
        left = random.randint(0, w - self.occlusion_size)

        # Apply the occlusion
        occluded_img[:, top:top+self.occlusion_size, left:left+self.occlusion_size] = 0
        return occluded_img, img


def get_dataloaders(config: Config):
    """
    Prepares MNIST dataloaders with a custom occlusion transform.
    The dataloader will yield tuples of (occluded_image, original_image, label).
    """
    base_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Custom dataset wrapper to apply occlusion and flattening
    class OccludedMNIST(MNIST):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.occlusion_transform = OcclusionTransform(config.occlusion_size)
            self.flatten_transform = transforms.Lambda(lambda x: x.view(-1))

        def __getitem__(self, index):
            img, label = super().__getitem__(index) # img is PIL

            # Apply base transform to get original tensor
            original_tensor = base_transform(img)

            # Apply occlusion
            occluded_tensor, _ = self.occlusion_transform(original_tensor)

            # Flatten images
            flat_occluded = self.flatten_transform(occluded_tensor)
            flat_original = self.flatten_transform(original_tensor)

            return flat_occluded, flat_original, label

    # Load the datasets
    train_val_dataset = OccludedMNIST(root="./data", train=True, download=True)
    test_dataset = OccludedMNIST(root="./data", train=False, download=True)

    # Split training set into training and validation
    train_size = int(0.9 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"DataLoaders created:")
    print(f"  - Train: {len(train_dataset)} samples")
    print(f"  - Validation: {len(val_dataset)} samples")
    print(f"  - Test: {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader

# --- Model Architectures ---

class BaselineAutoencoder(nn.Module):
    """A standard fully-connected autoencoder to serve as a baseline."""
    def __init__(self, layer_dims):
        super().__init__()
        # Encoder
        encoder_layers = []
        for i in range(len(layer_dims) - 1):
            encoder_layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        reversed_dims = layer_dims[::-1]
        for i in range(len(reversed_dims) - 1):
            decoder_layers.append(nn.Linear(reversed_dims[i], reversed_dims[i+1]))
            decoder_layers.append(nn.ReLU())
        # Remove the last ReLU and add a Sigmoid to output pixel values between 0 and 1
        self.decoder = nn.Sequential(*decoder_layers[:-1], nn.Sigmoid())

    def forward(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        # For the ablation study, we don't need a classifier head on the baseline.
        # We return a tuple to match the GIN's output signature.
        return reconstruction, None


class FCPredictiveLayer(nn.Module):
    """A stateless fully-connected predictive coding layer."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Bottom-up weights for propagating errors
        self.bu_weights = nn.Linear(input_dim, output_dim, bias=False)
        # Top-down weights for generating predictions
        self.td_weights = nn.Linear(output_dim, input_dim, bias=False)
        self.activation = nn.ReLU()

    def forward(self, *args, **kwargs):
        # This layer is stateless; its logic is handled in the main GIN class
        raise NotImplementedError("FCPredictiveLayer is stateless and shouldn't be called directly.")


class FCGIN(nn.Module):
    """A fully-connected Generative Inference Network."""
    def __init__(self, layer_dims, config: Config):
        super().__init__()
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims)
        self.config = config

        # Create the hierarchy of predictive layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.layers.append(FCPredictiveLayer(layer_dims[i], layer_dims[i+1]))

        # A simple classifier on top of the highest representation
        self.classifier = nn.Linear(self.layer_dims[-1], 10) # 10 classes for MNIST

    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device

        # --- 1. Initialize dynamic states (not part of the model's parameters) ---
        # Pyramidal (representation) states
        p_states = [torch.zeros(batch_size, dim, device=device) for dim in self.layer_dims]
        # Error states
        e_states = [torch.zeros(batch_size, dim, device=device) for dim in self.layer_dims]

        # --- 2. Inference Loop ---
        # The input image clamps the state of the lowest layer's error neurons
        e_states[0] = x - torch.sigmoid(self.layers[0].td_weights(p_states[1]))

        for t in range(self.config.inference_steps):
            # --- Top-down predictions ---
            for l in range(self.num_layers - 1, 0, -1):
                # Predict the state of the layer below
                prediction = self.layers[l-1].td_weights(p_states[l])

                # Update the error of the layer below
                # Note: For l=1, this updates the error for the input layer
                e_states[l-1] = p_states[l-1] - self.layers[l-1].activation(prediction)

            # --- Bottom-up error propagation ---
            for l in range(self.num_layers - 1):
                # Project error to the layer above
                error_proj = self.layers[l].bu_weights(e_states[l])

                # Update the representation state of the layer above
                # This is the core update rule: states change to reduce prediction error
                delta_p = self.config.gin_lr_inference * (error_proj - p_states[l+1])
                p_states[l+1] = p_states[l+1] + delta_p

        # --- 3. Final Outputs ---
        # The reconstruction is the final top-down prediction for the input layer
        reconstruction = torch.sigmoid(self.layers[0].td_weights(p_states[1]))

        # The classification is based on the final state of the top layer
        classification_logits = self.classifier(p_states[-1])

        return reconstruction, classification_logits


# --- Experiment Runner ---

def run_experiment(config: Config):
    """
    Runs a full training and evaluation experiment for a given configuration.
    """
    print(f"\n{'='*40}")
    print(f"🚀 Starting Experiment: {config.experiment_name}")
    print(f"{'='*40}")

    # --- Setup ---
    start_time = time.time()
    os.makedirs(f"results/{config.experiment_name}", exist_ok=True)

    device = torch.device(config.device)
    torch.manual_seed(42)

    # --- Data ---
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # --- Model ---
    if config.model_type == 'gin':
        model = FCGIN(config.layer_dims, config).to(device)
    elif config.model_type == 'autoencoder':
        model = BaselineAutoencoder(config.layer_dims).to(device)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    recon_criterion = nn.MSELoss()
    class_criterion = nn.CrossEntropyLoss()

    # --- Metrics ---
    accuracy_metric = MulticlassAccuracy(num_classes=10).to(device)

    # --- Training Loop ---
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_mse': [], 'val_acc': []}

    print(f"\nTraining {config.model_type} for {config.epochs} epochs on {device}...")
    for epoch in range(config.epochs):
        # Training
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            occluded_img, original_img, labels = [b.to(device) for b in batch]

            optimizer.zero_grad()
            recon, logits = model(occluded_img)

            loss_recon = recon_criterion(recon, original_img)

            if config.model_type == 'gin' and logits is not None:
                loss_class = class_criterion(logits, labels)
                loss = loss_recon + 0.1 * loss_class # Combine losses
            else:
                loss = loss_recon

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        total_val_mse = 0
        accuracy_metric.reset()
        with torch.no_grad():
            for batch in val_loader:
                occluded_img, original_img, labels = [b.to(device) for b in batch]
                recon, logits = model(occluded_img)

                loss_recon = recon_criterion(recon, original_img)
                total_val_mse += loss_recon.item()

                if config.model_type == 'gin' and logits is not None:
                    loss_class = class_criterion(logits, labels)
                    loss = loss_recon + 0.1 * loss_class
                    accuracy_metric.update(logits, labels)
                else:
                    loss = loss_recon

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_mse = total_val_mse / len(val_loader)
        val_acc = accuracy_metric.compute().item() if config.model_type == 'gin' else 0
        history['val_loss'].append(avg_val_loss)
        history['val_mse'].append(avg_val_mse)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{config.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val MSE: {avg_val_mse:.4f} | Val Acc: {val_acc:.3f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"results/{config.experiment_name}/best_model.pth")
            print(f"  -> New best model saved!")

    # --- Final Evaluation ---
    print("\n--- Evaluating on Test Set ---")
    model.load_state_dict(torch.load(f"results/{config.experiment_name}/best_model.pth"))
    model.eval()

    test_mse = 0
    test_ssim = 0
    test_lpips = 0
    accuracy_metric.reset()

    with torch.no_grad():
        for batch in test_loader:
            occluded_img, original_img, labels = [b.to(device) for b in batch]
            recon, logits = model(occluded_img)

            test_mse += recon_criterion(recon, original_img).item()

            # Reshape for image-based metrics
            recon_img = recon.view(-1, 1, 28, 28)
            original_img_reshaped = original_img.view(-1, 1, 28, 28)

            test_ssim += piq.ssim(recon_img, original_img_reshaped, data_range=1.).item()
            # LPIPS requires 3 channels, so we'll skip it for now to keep it simple
            # test_lpips += piq.lpips(recon_img.repeat(1,3,1,1), original_img_reshaped.repeat(1,3,1,1), reduction='mean').item()

            if config.model_type == 'gin' and logits is not None:
                accuracy_metric.update(logits, labels)

    final_mse = test_mse / len(test_loader)
    final_ssim = test_ssim / len(test_loader)
    final_acc = accuracy_metric.compute().item() if config.model_type == 'gin' else 0

    print(f"Final Test Metrics:")
    print(f"  - MSE: {final_mse:.6f}")
    print(f"  - SSIM: {final_ssim:.4f}")
    if config.model_type == 'gin':
        print(f"  - Accuracy: {final_acc:.4f}")

    # --- Visualization ---
    print("\n--- Generating Visualizations ---")
    occluded_vis, original_vis, _ = next(iter(test_loader))
    occluded_vis = occluded_vis[:config.num_vis_samples].to(device)
    original_vis = original_vis[:config.num_vis_samples].to(device)

    model.eval()
    with torch.no_grad():
        recon_vis, _ = model(occluded_vis)

    fig, axes = plt.subplots(3, config.num_vis_samples, figsize=(config.num_vis_samples * 1.5, 5))
    for i in range(config.num_vis_samples):
        axes[0, i].imshow(original_vis[i].cpu().numpy().reshape(28, 28), cmap='gray')
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')

        axes[1, i].imshow(occluded_vis[i].cpu().numpy().reshape(28, 28), cmap='gray')
        axes[1, i].set_title("Occluded")
        axes[1, i].axis('off')

        axes[2, i].imshow(recon_vis[i].cpu().numpy().reshape(28, 28), cmap='gray')
        axes[2, i].set_title("Recon")
        axes[2, i].axis('off')

    fig.suptitle(f"Reconstructions for {config.experiment_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"results/{config.experiment_name}/reconstructions.png")
    print(f"Saved reconstruction visualization to results/{config.experiment_name}/reconstructions.png")

    # --- Return Results ---
    end_time = time.time()
    print(f"\nExperiment finished in {end_time - start_time:.2f} seconds.")

    results = {
        "mse": final_mse,
        "ssim": final_ssim,
        "accuracy": final_acc
    }
    return results

# This block orchestrates the full ablation study
if __name__ == '__main__':
    # --- Define Experiment Configurations ---
    # To make this runnable in a short time, we'll use fewer epochs.
    # For a real study, epochs=20 or more would be better.
    shared_epochs = 3

    configs = [
        Config(
            model_type='autoencoder',
            experiment_name='baseline_autoencoder',
            epochs=shared_epochs
        ),
        Config(
            model_type='gin',
            experiment_name='gin_T2_inference',
            inference_steps=2,
            epochs=shared_epochs
        ),
        Config(
            model_type='gin',
            experiment_name='gin_T8_inference',
            inference_steps=8,
            epochs=shared_epochs
        ),
    ]

    # --- Run Experiments ---
    all_results = {}
    for config in configs:
        results = run_experiment(config)
        all_results[config.experiment_name] = results

    # --- Summarize Results ---
    print("\n\n" + "="*50)
    print("🔬 Final Ablation Study Results 🔬")
    print("="*50)

    # Header
    print(f"{'Experiment':<25} | {'MSE':<10} | {'SSIM':<10} | {'Accuracy':<10}")
    print("-" * 60)

    # Rows
    for name, metrics in all_results.items():
        print(f"{name:<25} | {metrics['mse']:<10.6f} | {metrics['ssim']:<10.4f} | {metrics['accuracy']:<10.4f}")

    print("="*50)

    # --- Combined Visualization ---
    print("\n--- Generating Combined Visualization ---")

    # Get a fixed batch of test data
    test_loader = get_dataloaders(Config())[2] # Just need the test loader
    occluded_vis, original_vis, _ = next(iter(test_loader))
    occluded_vis = occluded_vis[:configs[0].num_vis_samples].to(torch.device(configs[0].device))
    original_vis = original_vis[:configs[0].num_vis_samples].to(torch.device(configs[0].device))

    num_models = len(configs)
    num_samples = configs[0].num_vis_samples

    fig, axes = plt.subplots(num_models + 2, num_samples, figsize=(num_samples * 1.5, (num_models + 2) * 1.7))

    # Plot original and occluded images first
    for i in range(num_samples):
        axes[0, i].imshow(original_vis[i].cpu().numpy().reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title("Original", rotation=90, x=-0.1, y=0, va='center', ha='right', fontsize=12)

        axes[1, i].imshow(occluded_vis[i].cpu().numpy().reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title("Occluded", rotation=90, x=-0.1, y=0, va='center', ha='right', fontsize=12)

    # Plot reconstructions for each model
    for model_idx, config in enumerate(configs):
        # Load the best model for this config
        if config.model_type == 'gin':
            model = FCGIN(config.layer_dims, config)
        else:
            model = BaselineAutoencoder(config.layer_dims)
        model.load_state_dict(torch.load(f"results/{config.experiment_name}/best_model.pth"))
        model.to(torch.device(config.device))
        model.eval()

        with torch.no_grad():
            recon_vis, _ = model(occluded_vis)

        for i in range(num_samples):
            ax = axes[model_idx + 2, i]
            ax.imshow(recon_vis[i].cpu().numpy().reshape(28, 28), cmap='gray')
            ax.axis('off')
            if i == 0: ax.set_title(config.experiment_name, rotation=90, x=-0.1, y=0, va='center', ha='right', fontsize=8)

    fig.suptitle("Side-by-Side Model Reconstructions", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("ablation_study_summary.png")
    print("Saved combined visualization to ablation_study_summary.png")

    print("\n--- Script Finished ---")
