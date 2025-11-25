import torch
import matplotlib.pyplot as plt
import io
from PIL import Image

def render_token_importance(tokenizer, input_ids, mask=None, width=800, font_size=12):
    tokens = []
    for tid in input_ids:
        if tid == 0: continue 
        text = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        tokens.append(text)
    
    seq_len = len(tokens)
    if mask is not None: mask = mask[:seq_len]

    fig, ax = plt.subplots(figsize=(width/100, len(tokens)*0.5/5 + 2)) 
    ax.set_axis_off()
    
    x, y = 0.02, 0.90
    line_height = 0.15
    max_line_chars = 60
    current_line_chars = 0
    
    for i, token_text in enumerate(tokens):
        bg_color = (1, 1, 1, 0)
        text_color = 'black'
        alpha = 0.0
        
        if mask is not None:
            if mask[i] > 0.5:
                bg_color = 'red' 
                alpha = 0.3 
            else:
                text_color = 'blue'
                alpha = 0.5

        ax.text(x, y, token_text, fontsize=font_size, color=text_color, 
                bbox=dict(facecolor=bg_color, alpha=alpha, edgecolor='none', pad=2.0))
        
        char_len = len(token_text) + 1
        current_line_chars += char_len
        x += char_len * 0.018 
        
        if current_line_chars > max_line_chars:
            x, y = 0.02, y - line_height
            current_line_chars = 0
            
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)