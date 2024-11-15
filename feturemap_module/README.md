
# Feturemap Module

**特徵圖提取模組**  
本模組用於在訓練或預測時提取指定層的特徵圖，幫助分析模型的內部運作。

---

## 目錄

- [Feturemap Module](#feturemap-module)
  - [目錄](#目錄)
  - [模組環境建置說明](#模組環境建置說明)
  - [模組嵌入位置範例](#模組嵌入位置範例)
  - [訓練時提取特徵](#訓練時提取特徵)
  - [預測時提取特徵](#預測時提取特徵)
  - [修改想要看的層](#修改想要看的層)

---

## 模組環境建置說明

為了保證兼容性，**建議在虛擬環境中運行（例如 virtualenv）**。以下是建置環境的步驟：

1. 安裝 YOLOv11 的核心模組依賴：
   ```bash
   pip install ultralytics
   ```
2. 安裝完成後，移除 `ultralytics` 本身，但保留所有依賴：
   ```bash
   pip uninstall ultralytics
   ```
3. 克隆 YOLOv11 的官方代碼庫：
   ```bash
   git clone https://github.com/ultralytics/ultralytics.git
   ```
4. 將 `ultralytics` 文件夾移動到你的專案資料夾中，結構如下：

    ```plaintext
    ultralytics/
    ├── .github
    ├── docker
    ├── docs
    ├── examples
    ├── tests
    ├── ultralytics  # 取出這個放到你的專案資料夾下
    ├── README.md
    └── README.zh-CN.md
    ```

---

## 模組嵌入位置範例

以下是將模組嵌入 YOLOv11 專案的目錄結構範例：

```plaintext
你的專案資料夾/
├── ultralytics/
│   ├── assets
│   ├── cfg
│   ├── data
│   ├── engine
│   ├── feturemap_module/  # 特徵圖提取模組
│   │   ├── hooks_manager_predict.py  # 預測用特徵圖提取
│   │   ├── hooks_manager_train.py    # 訓練用特徵圖提取
│   └── __init__.py
│
├── train.py    # 訓練範例
└── predict.py  # 預測範例
```

---

## 訓練時提取特徵

`hooks_manager_train.py` 用於在訓練過程中提取指定層的特徵圖，支持以下功能：
- 支持多層特徵提取。
- 可選擇保存到本地或直接可視化。


使用方法範例：

**修改ultralytics/engine/trainer.py**

```python
from ultralytics.feturemap_module.hooks_manager_train import HooksManager_train
# 其他import
...
# 找到_do_train
def _do_train(self, world_size=1):
    if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)
    # 初始化 HooksManager_train
    hooks_manager = HooksManager_train(self.model,  self.save_dir, interval_epochs=50)

    ...

    epoch = self.start_epoch
    self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
    while True:
        self.epoch = epoch
        self.run_callbacks("on_train_epoch_start")
        if epoch % hooks_manager.interval_epochs == 0:
            hooks_manager.register_hooks()

        ...

        for i, batch in pbar:
            self.run_callbacks("on_train_batch_start")
            # Warmup

            ...
            #在for 迴圈內
            if epoch % hooks_manager.interval_epochs == 0:
                hooks_manager.save_feature_maps(epoch, i)
                hooks_manager.save_training_images(batch, self.loss, epoch, i)
                hooks_manager.clear_outputs()
                hooks_manager.remove_hooks()

```

---

## 預測時提取特徵

`hooks_manager_predict.py` 用於在推理階段提取特徵圖，適合用於模型分析與可視化。

使用方法範例：
例如我要使用Segment的Predictor
```python
from ultralytics.feturemap_module.hooks_manager_predict import HooksManager_predict
# 其他模組
...

class SegmentationPredictor(DetectionPredictor):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes the SegmentationPredictor with the provided configuration, overrides, and callbacks."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segment"
        self.hooks_manager = None  # 加入這行
    
    def setup_model(self, model, verbose=True):
        """Sets up the model and initializes hooks after model loading."""
        super().setup_model(model, verbose)
        # 確保模型載入後在初始化 HooksManager_predict
        if self.hooks_manager is None:
            self.hooks_manager = HooksManager_predict(self.model, self.save_dir)
            self.hooks_manager.register_hooks()

    def postprocess(self, preds, img, orig_imgs):
        
        ...

        for i, (pred, orig_img, img_path) in enumerate(zip(p, orig_imgs, self.batch[0])):
            if not len(pred):  # save empty boxes
                masks = None
                
            ...

            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks))

        #在最後放上特徵提取
        if self.hooks_manager:
            self.hooks_manager.save_feature_maps(self.save_dir, batch_idx=self.seen)
            self.hooks_manager.clear_outputs()
        return results
```

---

## 修改想要看的層

在hooks_manager_train.py、hooks_manager_predict.py中的__init__可以找到
[點擊查看 model_layer.txt](feturemap_module/model_layer.txt)

```python
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
```
