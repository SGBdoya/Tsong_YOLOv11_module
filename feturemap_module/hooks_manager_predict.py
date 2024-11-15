import os
import matplotlib.pyplot as plt

class HooksManager_predict:
    def __init__(self, model, save_dir):
        self.model = model
        self.save_dir = save_dir
        self.hooks = []
        self.outputs = {}
        # 指定要捕捉的層
        self.layers_to_capture = [
            "model.model.0",
            "model.model.0.conv",
            "model.model.1.conv",
            "model.model.2.m.0.cv2",
            "model.model.3",
            "model.model.9",
            "model.model.10",
            "model.model.11"
        ]


    def register_hooks(self):
        """Registers forward hooks to specified layers in the model."""
        for name, layer in self.model.named_modules():
            #print(f"Available layer: {name}")  # 新增除錯資訊
            if name in self.layers_to_capture:
                #print(f"Registering hook for layer: {name}")  # 除錯輸出
                hook = layer.register_forward_hook(self._hook_fn(name))
                self.hooks.append(hook)


    def _hook_fn(self, layer_name):
        def hook(module, input, output):
            # 儲存每一層的輸出為 tensor 格式
            #print(f"Capturing feature map for layer: {layer_name}")  # 除錯輸出
            self.outputs[layer_name] = output.clone().detach()
        return hook

    def save_feature_maps(self, save_dir, batch_idx):
        """Only saves feature maps if current batch matches the interval."""
        for layer_name, features in self.outputs.items():
            feature_path = os.path.join(
                save_dir, 'feature_maps', layer_name, f'batch_{batch_idx}')
            os.makedirs(feature_path, exist_ok=True)
            print(f"Saving feature maps for layer: {layer_name} to {feature_path}")  # 除錯輸出
            for j in range(features.shape[1]):
                plt.imshow(features[0, j].cpu().numpy(), cmap='gray')
                plt.title(f"{layer_name} - Channel {j}")
                plt.axis('off')
                plt.savefig(os.path.join(feature_path, f"{layer_name}_channel_{j}.png"), bbox_inches='tight')
                plt.close()

    def clear_outputs(self):
        """Clears the stored outputs after saving."""
        print("Clearing feature maps")  # 除錯輸出
        self.outputs.clear()

    def remove_hooks(self):
        """Removes all registered hooks."""
        print("Removing hooks")  # 除錯輸出
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
