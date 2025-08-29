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
try:
    import piq
    from torchmetrics.classification import MulticlassAccuracy
except ImportError:
    print("Please install piq and torchmetrics: pip install piq torchmetrics")
    exit()


@dataclass
class Config:
    # --- Experiment ---
    model_type: str = "gin"
    experiment_name: str = "gin_t8"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Data ---
    image_size: int = 28 * 28
    occlusion_size: int = 10

    # --- Model ---
    layer_dims: tuple = (image_size, 256, 128)
    use_layernorm: bool = False

    # --- GIN Specific ---
    inference_steps: int = 8
    gin_lr_inference: float = 0.1

    # --- Training ---
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 0.0

    # --- Visualization ---
    num_vis_samples: int = 10


class OcclusionTransform:
    def __init__(self, occlusion_size=10):
        self.occlusion_size = occlusion_size

    def __call__(self, img):
        c, h, w = img.shape
        occluded_img = img.clone()
        top = random.randint(0, h - self.occlusion_size)
        left = random.randint(0, w - self.occlusion_size)
        occluded_img[:, top:top+self.occlusion_size, left:left+self.occlusion_size] = 0
        return occluded_img, img


def get_dataloaders(config: Config):
    base_transform = transforms.Compose([transforms.ToTensor()])

    class OccludedMNIST(MNIST):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.occlusion_transform = OcclusionTransform(config.occlusion_size)
            self.flatten_transform = transforms.Lambda(lambda x: x.view(-1))

        def __getitem__(self, index):
            img, label = super().__getitem__(index)
            original_tensor = base_transform(img)
            occluded_tensor, _ = self.occlusion_transform(original_tensor)
            flat_occluded = self.flatten_transform(occluded_tensor)
            flat_original = self.flatten_transform(original_tensor)
            return flat_occluded, flat_original, label

    train_val_dataset = OccludedMNIST(root="./data", train=True, download=True)
    test_dataset = OccludedMNIST(root="./data", train=False, download=True)
    train_size = int(0.9 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader

# --- Model Architectures ---

class BaselineAutoencoder(nn.Module):
    def __init__(self, layer_dims):
        super().__init__()
        encoder_layers = []
        for i in range(len(layer_dims) - 1):
            encoder_layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)
        decoder_layers = []
        reversed_dims = layer_dims[::-1]
        for i in range(len(reversed_dims) - 1):
            decoder_layers.append(nn.Linear(reversed_dims[i], reversed_dims[i+1]))
            decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers[:-1], nn.Sigmoid())

    def forward(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, None

class FCPredictiveLayer(nn.Module):
    def __init__(self, input_dim, output_dim, use_layernorm=False):
        super().__init__()
        self.bu_weights = nn.Linear(input_dim, output_dim, bias=False)
        self.td_weights = nn.Linear(output_dim, input_dim, bias=False)
        self.activation = nn.ReLU()
        self.use_layernorm = use_layernorm
        if self.use_layernorm:
            self.ln_bu = nn.LayerNorm(output_dim)
            self.ln_td = nn.LayerNorm(input_dim)

class FCGIN(nn.Module):
    def __init__(self, layer_dims, config: Config):
        super().__init__()
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims)
        self.config = config
        self.layers = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.layers.append(FCPredictiveLayer(layer_dims[i], layer_dims[i+1], config.use_layernorm))
        self.classifier = nn.Linear(self.layer_dims[-1], 10)

    def forward(self, x):
        # --- Encoder Pass (Bottom-Up) ---
        h = x
        for layer in self.layers:
            h = layer.bu_weights(h)
            if self.config.use_layernorm:
                h = layer.ln_bu(h)
            h = layer.activation(h)
        p_state_final = h

        # --- Decoder Pass (Top-Down) ---
        h = p_state_final
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            h = layer.td_weights(h)
            if i > 0: # No activation or LN on the output layer
                if self.config.use_layernorm:
                    h = layer.ln_td(h)
                h = layer.activation(h)

        reconstruction = torch.sigmoid(h)
        classification_logits = self.classifier(p_state_final)
        return reconstruction, classification_logits

    def predictive_completion(self, x):
        batch_size = x.shape[0]
        device = x.device
        p_states = [torch.zeros(batch_size, dim, device=device) for dim in self.layer_dims]
        e_states = [torch.zeros(batch_size, dim, device=device) for dim in self.layer_dims]
        with torch.no_grad():
            h = x
            p_states[0] = x
            for i in range(len(self.layers)):
                h = self.layers[i].bu_weights(h)
                if self.config.use_layernorm:
                    h = self.layers[i].ln_bu(h)
                h = self.layers[i].activation(h)
                p_states[i+1] = h
        for t in range(self.config.inference_steps):
            for l in range(self.num_layers - 1, 0, -1):
                prediction = self.layers[l-1].td_weights(p_states[l])
                if self.config.use_layernorm:
                    prediction = self.layers[l-1].ln_td(prediction)
                e_states[l-1] = p_states[l-1] - self.layers[l-1].activation(prediction)
            for l in range(self.num_layers - 1):
                error_proj = self.layers[l].bu_weights(e_states[l])
                if self.config.use_layernorm:
                    error_proj = self.layers[l].ln_bu(error_proj)
                delta_p = self.config.gin_lr_inference * (error_proj - p_states[l+1])
                p_states[l+1] = p_states[l+1] + delta_p
        final_prediction = self.layers[0].td_weights(p_states[1])
        if self.config.use_layernorm:
            final_prediction = self.layers[0].ln_td(final_prediction)
        reconstruction = torch.sigmoid(final_prediction)
        classification_logits = self.classifier(p_states[-1])
        return reconstruction, classification_logits

def run_experiment(config: Config):
    print(f"\n{'='*40}")
    print(f"🚀 Starting Experiment: {config.experiment_name}")
    print(f"{'='*40}")

    start_time = time.time()
    os.makedirs(f"results/{config.experiment_name}", exist_ok=True)
    device = torch.device(config.device)
    torch.manual_seed(42)

    train_loader, val_loader, test_loader = get_dataloaders(config)

    if config.model_type == 'gin':
        model = FCGIN(config.layer_dims, config).to(device)
    elif config.model_type == 'autoencoder':
        model = BaselineAutoencoder(config.layer_dims).to(device)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    recon_criterion = nn.MSELoss()
    class_criterion = nn.CrossEntropyLoss()
    accuracy_metric = MulticlassAccuracy(num_classes=10).to(device)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_mse': [], 'val_acc': []}

    print(f"\nTraining {config.model_type} for {config.epochs} epochs on {device}...")
    for epoch in range(config.epochs):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            occluded_img, original_img, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            recon, logits = model(occluded_img)
            loss_recon = recon_criterion(recon, original_img)
            if config.model_type == 'gin' and logits is not None:
                loss_class = class_criterion(logits, labels)
                loss = loss_recon + loss_class
            else:
                loss = loss_recon
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        total_val_mse = 0
        accuracy_metric.reset()
        with torch.no_grad():
            for batch in val_loader:
                occluded_img, original_img, labels = [b.to(device) for b in batch]
                recon, logits = model(occluded_img)
                loss_recon = recon_criterion(recon, original_img)
                if config.model_type == 'gin' and logits is not None:
                    loss_class = class_criterion(logits, labels)
                    loss = loss_recon + loss_class
                    accuracy_metric.update(logits, labels)
                else:
                    loss = loss_recon
                total_val_mse += loss_recon.item()
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_mse = total_val_mse / len(val_loader)
        val_acc = accuracy_metric.compute().item() if config.model_type == 'gin' else 0

        print(f"Epoch {epoch+1}/{config.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val MSE: {avg_val_mse:.4f} | Val Acc: {val_acc:.3f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"results/{config.experiment_name}/best_model.pth")
            print(f"  -> New best model saved!")

    print("\n--- Evaluating on Test Set ---")
    model.load_state_dict(torch.load(f"results/{config.experiment_name}/best_model.pth"))
    model.eval()

    test_mse, test_ssim = 0, 0
    accuracy_metric.reset()
    with torch.no_grad():
        for batch in test_loader:
            occluded_img, original_img, labels = [b.to(device) for b in batch]
            recon, logits = model(occluded_img)
            test_mse += recon_criterion(recon, original_img).item()
            recon_img = recon.view(-1, 1, 28, 28)
            original_img_reshaped = original_img.view(-1, 1, 28, 28)
            test_ssim += piq.ssim(recon_img, original_img_reshaped, data_range=1.).item()
            if config.model_type == 'gin' and logits is not None:
                accuracy_metric.update(logits, labels)

    final_mse = test_mse / len(test_loader)
    final_ssim = test_ssim / len(test_loader)
    final_acc = accuracy_metric.compute().item() if config.model_type == 'gin' else 0

    print(f"Final Test Metrics (using standard forward pass):\n  - MSE: {final_mse:.6f}\n  - SSIM: {final_ssim:.4f}")
    if config.model_type == 'gin': print(f"  - Accuracy: {final_acc:.4f}")

    results = {"mse": final_mse, "ssim": final_ssim, "accuracy": final_acc}

    if config.model_type == 'gin':
        print("\n--- Evaluating GIN's Predictive Completion method ---")
        test_mse_pc = 0
        with torch.no_grad():
            for batch in test_loader:
                occluded_img, original_img, _ = [b.to(device) for b in batch]
                recon, _ = model.predictive_completion(occluded_img)
                test_mse_pc += recon_criterion(recon, original_img).item()
        final_mse_pc = test_mse_pc / len(test_loader)
        print(f"  - Predictive Completion MSE: {final_mse_pc:.6f}")
        results["mse_predictive_completion"] = final_mse_pc

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
        axes[0, i].set_title("Original"); axes[0, i].axis('off')
        axes[1, i].imshow(occluded_vis[i].cpu().numpy().reshape(28, 28), cmap='gray')
        axes[1, i].set_title("Occluded"); axes[1, i].axis('off')
        axes[2, i].imshow(recon_vis[i].cpu().numpy().reshape(28, 28), cmap='gray')
        axes[2, i].set_title("Recon"); axes[2, i].axis('off')
    fig.suptitle(f"Reconstructions for {config.experiment_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"results/{config.experiment_name}/reconstructions.png")

    end_time = time.time()
    print(f"Saved reconstruction visualization to results/{config.experiment_name}/reconstructions.png")
    print(f"\nExperiment finished in {end_time - start_time:.2f} seconds.")
    return results

if __name__ == '__main__':
    final_configs = [
        Config(
            model_type='autoencoder',
            experiment_name='corrected_baseline_autoencoder',
            epochs=5
        ),
        Config(
            model_type='gin',
            experiment_name='corrected_best_gin',
            inference_steps=8,
            epochs=5,
            learning_rate=1e-3,
            use_layernorm=True,
            weight_decay=1e-4
        ),
    ]

    all_results = {}
    for config in final_configs:
        results = run_experiment(config)
        all_results[config.experiment_name] = results

    print("\n\n" + "="*50)
    print("🔬 Final Corrected Ablation Study Results 🔬")
    print("="*50)

    header = f"{'Experiment':<35} | {'MSE (Train)':<12} | {'MSE (Predict)':<12} | {'SSIM':<10} | {'Accuracy':<10}"
    print(header)
    print("-" * len(header))

    for name, metrics in all_results.items():
        mse_pred = metrics.get('mse_predictive_completion', float('nan'))
        print(f"{name:<35} | {metrics['mse']:<12.6f} | {mse_pred:<12.6f} | {metrics['ssim']:<10.4f} | {metrics['accuracy']:<10.4f}")

    print("="*50)

    print("\n--- Script Finished ---")
