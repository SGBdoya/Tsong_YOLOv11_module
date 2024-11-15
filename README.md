# YOLOv11 Module

**嵌入式YOLOv11模組**
分析 YOLOv11 為目的製作的模組，包含特徵圖處理和視覺化工具。

---

## 模組目錄

1. [feturemap_module](#feturemap_module/README.md)
2. [grad_cam](#grad_cam)

---

## 簡介

YOLOv11 網路的特徵圖處理和視覺化分析功能。

---

## 模組概覽


| 模組名稱           | 功能簡介                                               |
| ------------------ | ------------------------------------------------------ |
| `feturemap_module` | 處理並可視化 YOLOv11 網路中的特徵圖                    |
| `grad_cam`         | 提供 Grad-CAM 視覺化，幫助解釋模型的預測結果和關注區域 |

---

## 嵌入位置範例

```plaintext
ultralytics/
├── assets
├── cfg
├── data
├── engine
...
├── feturemap_module/
│   ├── hooks_manager_predict.py # 預測用特徵圖提取
│   ├── hooks_manager.py # 訓練用特徵圖提取
└── __init__.py
```
