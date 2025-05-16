#!/usr/bin/env python
# gradcam_qwen_debug.py
# ---------------------------------------------------------------
import torch, matplotlib.pyplot as plt, numpy as np
from pathlib import Path
from PIL import Image
import cv2

from transformers import AutoProcessor, AutoModelForVision2Seq
from peft         import PeftModel
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch.nn as nn


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16                 # or bf16 on Hopper+


class ViTGradCAM(GradCAM):
    """
    Grad-CAM that works when the *input* is a token tensor [B, N, D].
    We just tell pytorch-grad-cam to use the H×W from grid_thw instead
    of guessing from the raw input shape.
    """
    def __init__(self, *args, grid_thw=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert grid_thw is not None, "grid_thw required"
        self._grid_thw = grid_thw            # torch.Tensor [B, 3]

    # override one function ↓
    def get_target_width_height(self, _input_tensor):
        h = int(self._grid_thw[0, 1])
        w = int(self._grid_thw[0, 2])
        return h, w

# ╭──────────────────────────────────────────────────────────────╮
# │  Vision-tower wrapper                                       │
# ╰──────────────────────────────────────────────────────────────╯
class VisionTowerWrapper(nn.Module):
    """
    Wrap the *full* Qwen-2.5-VL model so that:
        image  ➜  vision tower  ➜  CLS embedding
              ➜  language LM-head ➜  [B, vocab] logits
    Grad-CAM can then back-prop a chosen token logit.
    """
    def __init__(self, full_model):
        """
        full_model : Qwen2_5_VLForConditionalGeneration (or LoRA-merged PEFT)
        """
        super().__init__()
        self.full     = full_model                     # keep reference
        self.visual   = full_model.visual              # vision transformer
        self.lm_head  = full_model.lm_head             # text projection
        self.grid_thw = None                           # set once per batch

    # ────────────────────────────────────────────────────────────
    def forward(self, x):
        assert self.grid_thw is not None, "`grid_thw` must be set before forward()"

        # 1) Vision forward  → token embeddings  [B,N,D]  or  [B*N,D]
        vis_out = self.visual(x, self.grid_thw)
        hidden  = vis_out if isinstance(vis_out, torch.Tensor) else vis_out.last_hidden_state
        B       = self.grid_thw.shape[0]

        # un-flatten if flash-attn collapsed batch + tokens
        if hidden.dim() == 2:                          # [B*N, D] → [B,N,D]
            BN, D = hidden.shape
            N     = BN // B
            hidden = hidden.view(B, N, D)

        # 2) Choose CLS if present, else mean-pool tokens
        expected_tokens = 1 + int(self.grid_thw[0, 1] * self.grid_thw[0, 2])
        if hidden.shape[1] == expected_tokens:
            cls_emb = hidden[:, 0]                    # CLS token
        else:
            cls_emb = hidden.mean(dim=1)              # fallback

        # 3) Language projection  → logits [B, vocab]
        logits = self.lm_head(cls_emb)                # linear: D → V

        # 4) Ensure fp32 + grad so Grad-CAM can back-prop
        return logits.float().requires_grad_(True)




# ╭──────────────────────────────────────────────────────────────╮
# │  reshape_transform for ViT                                   │
# ╰──────────────────────────────────────────────────────────────╯
def reshape_vit(x, grid_thw):
    if isinstance(x, tuple): x = x[0]

    B = grid_thw.shape[0]
    h, w = map(int, grid_thw[0, 1:].tolist())
    D = x.shape[-1]

    # un-flatten if flash-attn collapsed batch+tokens
    if x.dim() == 2:
        BN = x.shape[0];  N = BN // B
        x  = x.view(B, N, D)
    elif x.dim() == 3:
        N = x.shape[1]
    else:
        raise ValueError(f"[reshape_vit] unexpected shape {x.shape}")

    ### DEBUG
    print(f"[reshape_vit] B={B}, N={N}, h={h}, w={w}, D={D}")

    # token layouts:
    #   (a) CLS + h*w tokens  → N == h*w+1
    #   (b)       h*w tokens  → N == h*w
    if N == h*w + 1:
        patches = x[:, 1:]
    elif N == h*w:
        patches = x
    else:
        raise AssertionError(
            f"Token/grid mismatch: got N={N} but h*w={h*w}"
        )                                               # ### ASSERT

    cam_in = patches.transpose(1, 2).reshape(B, D, h, w)

    ### DEBUG
    print(f"[reshape_vit] cam_in {cam_in.shape}")
    return cam_in


# ╭──────────────────────────────────────────────────────────────╮
# │  helpers                                                    │
# ╰──────────────────────────────────────────────────────────────╯
def load_models(lora_path):
    base = AutoModelForVision2Seq.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        device_map="auto", torch_dtype=DTYPE)
    tuned = PeftModel.from_pretrained(base, lora_path).merge_and_unload()
    return base, tuned

def vision_backbone(m):
    return (getattr(m, "visual", None) or
            getattr(getattr(m, "model", None), "visual", None) or
            getattr(m, "vision_tower", None))

def last_block(vit):
    return (vit.blocks[-1]            if hasattr(vit, "blocks")
            else vit.vision_model.encoder.layers[-1])

def preprocess(imgs, processor):
    batch = processor(images=imgs, text=[""]*len(imgs), return_tensors="pt")
    pixel = batch.pixel_values.to(DEVICE, DTYPE)
    grid  = batch.image_grid_thw.to(DEVICE)

    # ── NEW: ensure batch-axis is present ───────────────────────
    if pixel.dim() == 2:                      # [B*N, D] → [B, N, D]
        B = grid.shape[0]
        pixel = pixel.view(B, -1, pixel.size(-1))

    print(f"[preprocess] pixel {pixel.shape}, grid {grid}")
    return pixel, grid


def compute_cam(model, pixel, grid, processor):
    vit = vision_backbone(model)
    num_blocks = len(vit.blocks)        # Qwen-2.5-VL → 32
    print("Vision transformer has", num_blocks, "blocks")
    wrap = VisionTowerWrapper(model).to(DEVICE).eval()
    wrap.grid_thw = grid
    token_id = processor.tokenizer.encode("coffee", add_special_tokens=False)[0]
    targets  = [ClassifierOutputTarget(token_id) for _ in range(pixel.shape[0])]
    pixel_req = pixel.detach().clone().requires_grad_(True)

    layer_ids = [28,29,30,31]          # mix locality + semantics
    maps = []
    for i in layer_ids:
        cam = ViTGradCAM(model=wrap,
                        target_layers=[vit.blocks[i]],
                        reshape_transform=lambda o: reshape_vit(o, grid),
                        grid_thw=grid)
        maps.append(cam(pixel_req, targets=targets))
    mask = np.mean(maps, axis=0)    # (B, H, W)
    print(f"[GradCAM] masks {mask.shape}")
    return torch.from_numpy(mask)                   # ← convert once


def overlay(img_pil, mask):
    import numpy as np, cv2
    img_np = np.asarray(img_pil).astype(np.float32) / 255.0   # [H,W,3]

    # ── NEW: accept either torch.Tensor or np.ndarray ─────────────
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.squeeze(mask)          # drop extra channel if shape (1,H,W)

    H, W = img_np.shape[:2]
    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)

    mask = (mask - mask.min()) / (mask.ptp() + 1e-8)          # normalise
    return show_cam_on_image(img_np, mask, use_rgb=True, image_weight=0.5)



# ╭──────────────────────────────────────────────────────────────╮
# │  main                                                       │
# ╰──────────────────────────────────────────────────────────────╯
def main():
    print("GPU:", torch.cuda.get_device_name(), "| DTYPE:", DTYPE)

    images    = ["data/binaries/captured_images/captured_image_0.jpg"]
    lora_ckpt = "saves/qwen2.5_vl-7b/lora/sft/grocery_589824_res"

    base, tuned = load_models(lora_ckpt)
    processor   = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    imgs_pil          = [Image.open(p).convert("RGB") for p in images]
    pixel, grid       = preprocess(imgs_pil, processor)
    masks_base        = compute_cam(base,  pixel, grid, processor)
    masks_tuned       = compute_cam(tuned, pixel, grid, processor)

    # quick visual sanity-check
    fig, ax = plt.subplots(1,2, figsize=(8,4))
    for i, m in enumerate([masks_base[0], masks_tuned[0]]):   # m is tensor
        ax[i].imshow(overlay(imgs_pil[0], m))     
        ax[i].set_title(["BASE","TUNED"][i]); ax[i].axis("off")
    out = Path("plots/attention_mask.png"); plt.tight_layout(); fig.savefig(out, dpi=250)
    print("✅ wrote", out)

if __name__ == "__main__":
    main()
