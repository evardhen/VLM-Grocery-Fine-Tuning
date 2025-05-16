import torch, matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
import torch
from qwen_vl_utils import process_vision_info

DEVICE, DTYPE = "cuda", torch.float16
BASE_MODEL    = "Qwen/Qwen2.5-VL-7B-Instruct"
LORA_PATH   = "saves/qwen2.5_vl-7b/lora/sft/grocery_589824_res"
merge = True
if merge:
    processor     = AutoProcessor.from_pretrained(BASE_MODEL, device_map="auto")
    base          = AutoModelForVision2Seq.from_pretrained(BASE_MODEL,
                                                        device_map="auto",
                                                        torch_dtype=DTYPE)

    lora_model = PeftModel.from_pretrained(
        base,             # wrap the *already-loaded* base model
        LORA_PATH,        # path or hub-id of your adapter
        device_map="auto",
        torch_dtype=DTYPE,
    )

    model = lora_model.merge_and_unload()   # 🍰  weights baked in, no PEFT left
else:
    processor     = AutoProcessor.from_pretrained(BASE_MODEL, device_map="auto")
    model          = AutoModelForVision2Seq.from_pretrained(BASE_MODEL,
                                                        device_map="auto",
                                                        torch_dtype=DTYPE)
model.eval() 


def aggregate_vis_curve(attn, image_mask, layer="last"):
    """
    attn       : tuple[Tensor]  length L, each (1, H, S, S)
    image_mask : bool Tensor (S,)  True where token is an image token
    layer      : "last" | "mean"
    returns    : txt_rows (LongTensor), vis_curve (np.array)
    """
    img_cols = image_mask.nonzero(as_tuple=False).squeeze(-1)      # (N_img,)
    txt_rows = (~image_mask).nonzero(as_tuple=False).squeeze(-1)   # (N_txt,)

    if layer == "last":
        # ─── just the final layer ──────────────────────────────────────────
        A = attn[-1].mean(1).squeeze(0)                            # (S,S) on GPU
        vis_curve = A[txt_rows][:, img_cols].sum(-1).cpu().numpy()
        return txt_rows, vis_curve

    elif layer == "mean":
        # ─── streaming mean over layers (no big tensor) ───────────────────
        running = None
        for l, A in enumerate(attn, start=1):                      # iterate layers
            A_cpu = A.mean(1).squeeze(0).cpu()                    # → (S,S) on CPU
            if running is None:
                running = A_cpu
            else:                                                 # incremental mean
                running += (A_cpu - running) / l
        vis_curve = running[txt_rows.cpu()][:, img_cols.cpu()].sum(-1).numpy()
        return txt_rows, vis_curve

    else:
        raise ValueError("layer must be 'last' or 'mean'")


def clean_and_mask(raw_tokens):
    """
    raw_tokens : list[str]  (same length as `curve`)
    returns     : cleaned_tokens, keep_mask (torch.bool of same len)
    """
    n = len(raw_tokens)
    keep = torch.ones(n, dtype=torch.bool)

    # 1) drop *before* first <|im_end|>
    try:
        first_end = raw_tokens.index("<|vision_end|>")
        keep[: first_end + 1] = False
    except ValueError:
        pass

    # 2) drop *after* last <|im_end|>
    try:
        last_end = n - 1 - raw_tokens[::-1].index("<|im_end|>")
        keep[last_end :] = False
    except ValueError:
        pass

    special = {
        "<|im_start|>", "<|im_end|>", "<|endoftext|>",
        "<|image_pad|>", "<PAD>", "",
    }

    cleaned = []
    for i, tok in enumerate(raw_tokens):
        if tok == "Ġ":
            keep[i] = False
            continue
        if not keep[i]:                     # we already decided to drop
            continue
        if tok in special:                  # drop template token
            keep[i] = False
            continue
        # if tok == "Ċ":
        #     keep[i] = False
        #     continue
        elif tok.startswith("Ġ"):
            tok = tok[1:]
        cleaned.append(tok)

    return cleaned, keep                   # keep is a torch.bool mask



def token_to_label(tid):
    txt = processor.tokenizer.decode([tid])
    txt = txt.replace("\n", "\\n")        # show newline explicitly
    return txt.lstrip()                   # drop leading space


def tokens_and_vis_attn(img_path, user_prompt):
    # ------------------------------------------------------------
    # 1) build a *chat template* with an image slot
    # ------------------------------------------------------------

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},                       # <|image|> placeholder
                {"type": "text", "text": user_prompt},  # your words
            ],
        }
    ]

    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, _, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=chat_text, images=images, padding=True, return_tensors="pt")
    print(inputs)
    input_ids   = inputs.input_ids.to(DEVICE)                  # [1, N+T]
    grid        = inputs.image_grid_thw.to(DEVICE)             # [[1, Hp, Wp]]
    pixel_values  = inputs.pixel_values.to(DEVICE).to(DTYPE)     # [B*N, D]
    print(f"Pixel values shape: {pixel_values.shape}")
    print(f"Grid: {grid}")


    # ------------------------------------------------------------
    # 3) forward with attentions
    # ------------------------------------------------------------
    with torch.no_grad():
        out = model(
            pixel_values   = pixel_values,
            image_grid_thw = grid,
            input_ids      = input_ids,
            output_attentions=True,
            return_dict=True,
        )

    img_tok_id = model.config.image_token_id
    image_mask = (input_ids[0] == img_tok_id)
    txt_rows, curve = aggregate_vis_curve(
            out.attentions, image_mask, layer="mean")   # or "mean"
    tokens = processor.tokenizer.convert_ids_to_tokens(
                input_ids[0, txt_rows].tolist())
    
    # --- strip system-prompt section --------------------------------
    _, keep_mask = clean_and_mask(tokens)
    plot_curve  = curve[keep_mask.cpu().numpy()] 
    labels = [token_to_label(t) for t in input_ids[0, txt_rows][keep_mask]]

    return labels, plot_curve


# ------------------------------------------------------------------
# run it
# ------------------------------------------------------------------
prompt   = "Category: Tea\nFine-grained category: Fruit Tea\nCount: 1\n\nCategory: Tea\nFine-grained category: Fennel Anise Caraway Tea\nCount: 1\n\n"

tok, curve = tokens_and_vis_attn("data/binaries/captured_images/captured_image_195.jpg", prompt)

# ------------------------------------------------------------------
# plot
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(curve, marker ="o")
ax.set_xticks(range(len(tok)))
ax.set_xticklabels(tok, rotation=90, fontsize=8)
ax.set_xlabel("Generated Token", fontsize=14, labelpad=20)
ax.set_ylabel("Vision Attention (%)", fontsize=14, labelpad=12)
# ─── annotate specific tokens ──────────────────────────────────
targets = {"Tea", "1", "Fruit", "-gr", "el"}                 # tokens you care about
for idx, (lbl, y) in enumerate(zip(tok, curve)):
    if lbl in targets:                       # the tokens you care about
        ax.annotate(f"{y:.3f}",
                    xy=(idx, y),                  # anchor at the data point
                    xytext=(0, 6),                  # offset (pixels)
                    textcoords="offset points",   # interpret (0,10) in pts
                    ha="center", va="bottom",
                    fontsize=8, color="tab:red",
                    bbox=dict(facecolor="white",  # tiny white box
                              edgecolor="none",   #   that hides the line
                              alpha=0.8,          #   (slightly translucent)
                              pad=1.0))

plt.tight_layout()
if merge:
    plt.savefig("plots/vision_attention_tuned.png", dpi=300)
    print("✅  wrote plots/vision_attention_tuned.png")
else:
    plt.savefig("plots/vision_attention_base.png", dpi=300)
    print("✅  wrote plots/vision_attention_base.png")  
