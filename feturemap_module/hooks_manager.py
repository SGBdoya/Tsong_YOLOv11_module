import os
import matplotlib.pyplot as plt


class HooksManager:
    def __init__(self, model, save_dir, interval_epochs=50):
        self.model = model
        self.save_dir = save_dir
        self.interval_epochs = interval_epochs
        self.hooks = []
        self.outputs = {}
        # 指定要捕捉的層
        self.layers_to_capture = [
            "model.0",
            "model.0.conv",
            "model.1.conv",
            "model.2.m.0.cv2",
            "model.3",
            "model.9",
            "model.10",
            "model.11"
        ]

    def register_hooks(self):
        """Registers forward hooks to specified layers in the model."""
        for name, layer in self.model.named_modules():
            if name in self.layers_to_capture:
                hook = layer.register_forward_hook(self._hook_fn(name))
                self.hooks.append(hook)

    def _hook_fn(self, layer_name):
        def hook(module, input, output):
            # 儲存每一層的輸出為 tensor 格式
            self.outputs[layer_name] = output.clone().detach()
        return hook

    def save_feature_maps(self, epoch, batch_idx):
        """Only saves feature maps if current epoch matches the interval."""
        for layer_name, features in self.outputs.items():
            feature_path = os.path.join(
                self.save_dir, 'feature_maps', layer_name, f'epoch_{epoch}_batch_{batch_idx}')
            os.makedirs(feature_path, exist_ok=True)
            for j in range(features.shape[1]):
                plt.imshow(features[0, j].cpu().numpy(), cmap='gray')
                plt.title(f"{layer_name} - Channel {j}")
                plt.axis('off')
                plt.savefig(os.path.join(feature_path, f"{layer_name}_channel_{j}.png"), bbox_inches='tight')
                plt.close()

    def save_training_images(self, batch, predictions, epoch, batch_idx):
        """Only saves training images if current epoch matches the interval."""
        # 1. 儲存數據增強後的樣本
        aug_path = os.path.join(self.save_dir, 'augmented_samples', f'epoch_{epoch}_batch_{batch_idx}')
        os.makedirs(aug_path, exist_ok=True)
        for i, img in enumerate(batch["img"]):
            img = img.permute(1, 2, 0).cpu().numpy()
            plt.imshow(img)
            plt.title(f"Augmented Sample {i}")
            plt.axis('off')
            plt.savefig(os.path.join(aug_path, f"augmented_{i}.png"), bbox_inches='tight')
            plt.close()

        # 2. 儲存模型預測結果
        pred_path = os.path.join(self.save_dir, 'predictions', f'epoch_{epoch}_batch_{batch_idx}')
        os.makedirs(pred_path, exist_ok=True)
        for i, img in enumerate(batch["img"]):
            img = img.permute(1, 2, 0).cpu().numpy()
            plt.imshow(img)
            plt.title(f"Prediction {i}")
            plt.axis('off')
            plt.savefig(os.path.join(pred_path, f"prediction_{i}.png"), bbox_inches='tight')
            plt.close()

        # 3. 儲存啟用分佈直方圖
        for layer_name, features in self.outputs.items():
            act_hist_path = os.path.join(
                self.save_dir, 'activation_histograms', layer_name, f'epoch_{epoch}_batch_{batch_idx}')
            os.makedirs(act_hist_path, exist_ok=True)
            activation_values = features.cpu().numpy().flatten()
            plt.hist(activation_values, bins=50)
            plt.title(f"{layer_name} Activation Distribution")
            plt.xlabel("Activation Value")
            plt.ylabel("Frequency")
            plt.savefig(os.path.join(act_hist_path, f"{layer_name}_activation_histogram.png"), bbox_inches='tight')
            plt.close()

    def clear_outputs(self):
        """Clears the stored outputs after saving."""
        self.outputs.clear()

    def remove_hooks(self):
        """Removes all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

