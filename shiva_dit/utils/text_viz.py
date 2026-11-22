import matplotlib.pyplot as plt
import io
from PIL import Image


def render_token_importance(tokenizer, input_ids, mask=None, width=1000, font_size=10):
    processed_tokens = []

    if hasattr(input_ids, "cpu"):
        input_ids = input_ids.cpu().tolist()
    if mask is not None and hasattr(mask, "cpu"):
        mask = mask.cpu().tolist()

    current_pad_count = 0
    current_pad_mask_sum = 0.0

    for i, tid in enumerate(input_ids):
        is_pad = tid == 0
        token_mask_val = mask[i] if (mask is not None and i < len(mask)) else 0.0

        if is_pad:
            current_pad_count += 1
            current_pad_mask_sum += token_mask_val
        else:
            if current_pad_count > 0:
                avg_mask = current_pad_mask_sum / current_pad_count
                processed_tokens.append(
                    {
                        "text": f"[PAD x{current_pad_count}]",
                        "mask": avg_mask,
                        "type": "pad",
                    }
                )
                current_pad_count = 0
                current_pad_mask_sum = 0.0

            try:
                text = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
            except Exception:
                text = ""
            if not text.strip():
                text = f"[{tid}]"

            processed_tokens.append(
                {"text": text, "mask": token_mask_val, "type": "word"}
            )

    if current_pad_count > 0:
        avg_mask = current_pad_mask_sum / current_pad_count
        processed_tokens.append(
            {"text": f"[PAD x{current_pad_count}]", "mask": avg_mask, "type": "pad"}
        )

    CANVAS_WIDTH = 100
    x_cursor = 0
    y_cursor = 0
    layout_items = []

    char_width_factor = 1.2
    box_padding = 2.5

    for item in processed_tokens:
        text_len = len(item["text"])
        item_width = (text_len * char_width_factor) + box_padding

        if x_cursor + item_width > CANVAS_WIDTH:
            x_cursor = 0
            y_cursor += 1

        layout_items.append(
            {
                "x": x_cursor,
                "y": y_cursor,
                "text": item["text"],
                "mask": item["mask"],
                "type": item["type"],
            }
        )

        x_cursor += item_width

    total_rows = y_cursor + 1

    fig_height = max(2, total_rows * 0.6)
    fig, ax = plt.subplots(figsize=(12, fig_height), dpi=120)
    ax.set_xlim(0, CANVAS_WIDTH)
    ax.set_ylim(total_rows, -1)
    ax.set_axis_off()

    for item in layout_items:
        mask_val = item["mask"]
        text = item["text"]

        if mask_val > 0.5:
            facecolor = "#ffcccc"  # Light red background
            edgecolor = "red"
            textcolor = "black"
            alpha = 0.9
            linewidth = 1.5
        else:
            facecolor = "#e6f2ff"
            edgecolor = "blue"
            textcolor = "gray"
            alpha = 0.6
            linewidth = 0.5

        if item["type"] == "pad":
            text_style = "italic"
        else:
            text_style = "normal"

        ax.text(
            item["x"] + 1,
            item["y"],
            text,
            fontsize=font_size,
            color=textcolor,
            style=text_style,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=facecolor,
                edgecolor=edgecolor,
                alpha=alpha,
                linewidth=linewidth,
            ),
        )

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)
