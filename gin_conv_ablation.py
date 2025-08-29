# gin_conv_ablation.py
# A script to conduct an ablation study on a new CONVOLUTIONAL Generative Inference Network (GIN)
# for occlusion completion on the MNIST dataset.

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
    experiment_name: str = "conv_gin"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Data ---
    occlusion_size: int = 10

    # --- Model ---
    use_norm: bool = True

    # --- GIN Specific ---
    inference_steps: int = 8
    gin_lr_inference: float = 0.1

    # --- Training ---
    epochs: int = 5
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

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

        def __getitem__(self, index):
            img, label = super().__getitem__(index)
            original_tensor = base_transform(img)
            occluded_tensor, _ = self.occlusion_transform(original_tensor)
            return occluded_tensor, original_tensor, label

    train_val_dataset = OccludedMNIST(root="./data", train=True, download=True)
    test_dataset = OccludedMNIST(root="./data", train=False, download=True)
    train_size = int(0.9 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader

# --- Convolutional Model Architectures ---

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        self.classifier_head = nn.Sequential(nn.Flatten(), nn.Linear(32 * 7 * 7, 10))

    def forward(self, x):
        h = self.encoder(x)
        logits = self.classifier_head(h)
        reconstruction = self.decoder(h)
        return reconstruction, logits

class ConvPredictiveLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, use_norm=True):
        super().__init__()
        self.bu_weights = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, stride=2)
        self.td_weights = nn.ConvTranspose2d(out_channels, in_channels, kernel_size, padding=1, stride=2, output_padding=1)
        self.activation = nn.ReLU()
        self.use_norm = use_norm
        if self.use_norm:
            self.norm_bu = nn.GroupNorm(4, out_channels)
            self.norm_td = nn.GroupNorm(max(1, in_channels//4), in_channels)

class ConvGIN(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.channel_dims = [1, 16, 32]
        self.layers = nn.ModuleList()
        for i in range(len(self.channel_dims) - 1):
            self.layers.append(ConvPredictiveLayer(self.channel_dims[i], self.channel_dims[i+1], use_norm=config.use_norm))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32 * 7 * 7, 10))

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer.bu_weights(h)
            if self.config.use_norm: h = layer.norm_bu(h)
            h = layer.activation(h)
        p_state_final = h
        h = p_state_final
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            h = layer.td_weights(h)
            if i > 0:
                if self.config.use_norm: h = layer.norm_td(h)
                h = layer.activation(h)
        reconstruction = torch.sigmoid(h)
        logits = self.classifier(p_state_final)
        return reconstruction, logits

    def predictive_completion(self, x, mask):
        device = x.device
        p_states = [torch.zeros_like(x)]
        e_states = [torch.zeros_like(x)]
        with torch.no_grad():
            h = x
            for i, layer in enumerate(self.layers):
                h = layer.bu_weights(h)
                if self.config.use_norm: h = layer.norm_bu(h)
                h = layer.activation(h)
                p_states.append(h)
                e_states.append(torch.zeros_like(h))
        for _ in range(self.config.inference_steps):
            for l in range(len(self.layers) - 1, -1, -1):
                layer = self.layers[l]
                prediction = layer.td_weights(p_states[l+1])
                if self.config.use_norm: prediction = layer.norm_td(prediction)
                error = p_states[l] - layer.activation(prediction)
                if l == 0: e_states[l] = (x - layer.activation(prediction)) * mask
                else: e_states[l] = error
            for l in range(len(self.layers)):
                layer = self.layers[l]
                bu_projection = layer.bu_weights(e_states[l])
                if self.config.use_norm: bu_projection = layer.norm_bu(bu_projection)
                delta_p = self.config.gin_lr_inference * (bu_projection - e_states[l+1])
                p_states[l+1] = p_states[l+1] + delta_p
        h = p_states[-1]
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            h = layer.td_weights(h)
            if i > 0:
                if self.config.use_norm: h = layer.norm_td(h)
                h = layer.activation(h)
        reconstruction = torch.sigmoid(h)
        logits = self.classifier(p_states[-1])
        return reconstruction, logits

def run_experiment(config: Config):
    print(f"\n{'='*40}\n🚀 Starting Experiment: {config.experiment_name}\n{'='*40}")
    start_time = time.time()
    os.makedirs(f"results/{config.experiment_name}", exist_ok=True)
    device = torch.device(config.device)
    torch.manual_seed(42)
    train_loader, val_loader, test_loader = get_dataloaders(config)
    if config.model_type == 'gin': model = ConvGIN(config).to(device)
    elif config.model_type == 'autoencoder': model = ConvAutoencoder().to(device)
    else: raise ValueError(f"Unknown model_type: {config.model_type}")
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    recon_criterion, class_criterion = nn.MSELoss(), nn.CrossEntropyLoss()
    accuracy_metric = MulticlassAccuracy(num_classes=10).to(device)
    best_val_loss = float('inf')
    print(f"\nTraining {config.model_type} for {config.epochs} epochs on {device}...")
    for epoch in range(config.epochs):
        model.train()
        for occluded_img, original_img, labels in train_loader:
            occluded_img, original_img, labels = occluded_img.to(device), original_img.to(device), labels.to(device)
            optimizer.zero_grad()
            recon, logits = model(occluded_img)
            loss = recon_criterion(recon, original_img) + class_criterion(logits, labels)
            loss.backward(); optimizer.step()
        model.eval()
        total_val_loss = 0; accuracy_metric.reset()
        with torch.no_grad():
            for occluded_img, original_img, labels in val_loader:
                occluded_img, original_img, labels = occluded_img.to(device), original_img.to(device), labels.to(device)
                recon, logits = model(occluded_img)
                loss = recon_criterion(recon, original_img) + class_criterion(logits, labels)
                total_val_loss += loss.item(); accuracy_metric.update(logits, labels)
        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = accuracy_metric.compute().item()
        print(f"Epoch {epoch+1}/{config.epochs} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.3f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"results/{config.experiment_name}/best_model.pth")
            print(f"  -> New best model saved!")
    print("\n--- Evaluating on Test Set ---")
    model.load_state_dict(torch.load(f"results/{config.experiment_name}/best_model.pth"))
    model.eval()
    test_mse, test_ssim = 0, 0; accuracy_metric.reset()
    with torch.no_grad():
        for occluded_img, original_img, labels in test_loader:
            occluded_img, original_img, labels = occluded_img.to(device), original_img.to(device), labels.to(device)
            recon, logits = model(occluded_img)
            test_mse += recon_criterion(recon, original_img).item()
            test_ssim += piq.ssim(recon, original_img, data_range=1.).item()
            accuracy_metric.update(logits, labels)
    final_mse = test_mse / len(test_loader); final_ssim = test_ssim / len(test_loader)
    final_acc = accuracy_metric.compute().item()
    print(f"Final Test Metrics (standard forward pass):\n  - MSE: {final_mse:.6f}\n  - SSIM: {final_ssim:.4f}\n  - Accuracy: {final_acc:.4f}")
    results = {"mse": final_mse, "ssim": final_ssim, "accuracy": final_acc}
    if config.model_type == 'gin':
        print("\n--- Evaluating GIN's Predictive Completion method ---")
        test_mse_pc, test_ssim_pc = 0, 0; accuracy_metric.reset()
        with torch.no_grad():
            for occluded_img, original_img, labels in test_loader:
                occluded_img, original_img, labels = occluded_img.to(device), original_img.to(device), labels.to(device)
                mask = (occluded_img > 0).float()
                recon, logits = model.predictive_completion(occluded_img, mask)
                test_mse_pc += recon_criterion(recon, original_img).item()
                test_ssim_pc += piq.ssim(recon, original_img, data_range=1.).item()
                accuracy_metric.update(logits, labels)
        final_mse_pc = test_mse_pc / len(test_loader); final_ssim_pc = test_ssim_pc / len(test_loader)
        final_acc_pc = accuracy_metric.compute().item()
        print(f"Predictive Completion Metrics:\n  - MSE: {final_mse_pc:.6f}\n  - SSIM: {final_ssim_pc:.4f}\n  - Accuracy: {final_acc_pc:.4f}")
        results["mse_predictive"] = final_mse_pc; results["ssim_predictive"] = final_ssim_pc; results["acc_predictive"] = final_acc_pc
    print("\n--- Generating Visualizations ---")
    occluded_vis, original_vis, _ = next(iter(test_loader))
    occluded_vis = occluded_vis[:config.num_vis_samples].to(device)
    original_vis = original_vis[:config.num_vis_samples].to(device)
    model.eval()
    with torch.no_grad(): recon_vis, _ = model(occluded_vis)
    fig, axes = plt.subplots(3, config.num_vis_samples, figsize=(config.num_vis_samples * 1.5, 5))
    for i in range(config.num_vis_samples):
        axes[0, i].imshow(original_vis[i].cpu().squeeze(), cmap='gray'); axes[0, i].set_title("Original"); axes[0, i].axis('off')
        axes[1, i].imshow(occluded_vis[i].cpu().squeeze(), cmap='gray'); axes[1, i].set_title("Occluded"); axes[1, i].axis('off')
        axes[2, i].imshow(recon_vis[i].cpu().squeeze(), cmap='gray'); axes[2, i].set_title("Recon"); axes[2, i].axis('off')
    fig.suptitle(f"Reconstructions for {config.experiment_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(f"results/{config.experiment_name}/reconstructions.png")
    print(f"Saved reconstruction visualization to results/{config.experiment_name}/reconstructions.png")
    end_time = time.time()
    print(f"\nExperiment finished in {end_time - start_time:.2f} seconds.")
    return results

if __name__ == '__main__':
    final_configs = [
        Config(model_type='autoencoder', experiment_name='conv_autoencoder', epochs=5),
        Config(model_type='gin', experiment_name='conv_gin', epochs=5),
    ]
    all_results = {}
    for config in final_configs:
        results = run_experiment(config)
        all_results[config.experiment_name] = results
    print("\n\n" + "="*80)
    print("🔬 Final Convolutional Ablation Study Results 🔬".center(80))
    print("="*80)
    header = f"{'Experiment':<25} | {'MSE (Train)':<12} | {'MSE (Predict)':<12} | {'SSIM':<10} | {'Accuracy':<10}"
    print(header)
    print("-" * len(header))
    for name, metrics in all_results.items():
        mse_pred = metrics.get('mse_predictive', float('nan'))
        ssim_pred = metrics.get('ssim_predictive', float('nan'))
        acc_pred = metrics.get('acc_predictive', float('nan'))
        print(f"{name:<25} | {metrics['mse']:<12.6f} | {mse_pred:<12.6f} | {metrics['ssim']:<10.4f} | {metrics['accuracy']:<10.4f}")
    print("="*80)
    print("\n--- Generating Combined Final Visualization ---")
    try:
        test_loader = get_dataloaders(Config())[2]
        occluded_vis, original_vis, _ = next(iter(test_loader))
        occluded_vis = occluded_vis[:final_configs[0].num_vis_samples].to(torch.device(final_configs[0].device))
        original_vis = original_vis[:final_configs[0].num_vis_samples].to(torch.device(final_configs[0].device))
        fig, axes = plt.subplots(4, final_configs[0].num_vis_samples, figsize=(final_configs[0].num_vis_samples * 1.5, 8))
        for i in range(final_configs[0].num_vis_samples):
            axes[0, i].imshow(original_vis[i].cpu().squeeze(), cmap='gray'); axes[0, i].axis('off')
            if i == 0: axes[0, i].set_title("Original", rotation=90, x=-0.2, y=0, va='center', ha='right')
            axes[1, i].imshow(occluded_vis[i].cpu().squeeze(), cmap='gray'); axes[1, i].axis('off')
            if i == 0: axes[1, i].set_title("Occluded", rotation=90, x=-0.2, y=0, va='center', ha='right')

        for model_idx, config in enumerate(final_configs):
            model_path = f"results/{config.experiment_name}/best_model.pth"
            if not os.path.exists(model_path):
                print(f"Warning: Model file not found at {model_path}.")
                continue
            if config.model_type == 'gin': model = ConvGIN(config)
            else: model = ConvAutoencoder()
            model.load_state_dict(torch.load(model_path, map_location=config.device)); model.to(config.device); model.eval()
            with torch.no_grad():
                recon_vis, _ = model(occluded_vis)
                if config.model_type == 'gin':
                    mask = (occluded_vis > 0).float()
                    recon_pc, _ = model.predictive_completion(occluded_vis, mask)

            for i in range(final_configs[0].num_vis_samples):
                ax = axes[model_idx + 2, i]
                recon_to_show = recon_vis if model_idx == 0 else recon_pc if config.model_type == 'gin' else recon_vis
                ax.imshow(recon_to_show[i].cpu().squeeze(), cmap='gray'); ax.axis('off')
                title = f"{config.experiment_name}"
                if config.model_type == 'gin' and model_idx == 1: title += " (Predictive)"
                if i == 0: ax.set_title(title, rotation=90, x=-0.2, y=0, va='center', ha='right')
                if model_idx == 1 and config.model_type == 'gin' :
                     axes[2, i].imshow(recon_vis[i].cpu().squeeze(), cmap='gray'); axes[2, i].axis('off')
                     if i==0: axes[2,i].set_title("conv_gin (Standard)", rotation=90, x=-0.2, y=0, va='center', ha='right')


        fig.suptitle("Final Side-by-Side Model Reconstructions", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("final_conv_ablation_summary.png")
        print("Saved combined visualization to final_conv_ablation_summary.png")
    except Exception as e:
        print(f"\nCould not generate final visualization due to an error: {e}")
    print("\n--- Script Finished ---")
