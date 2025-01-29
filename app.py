import gradio as gr
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Load Model NLP Ringan (DistilBERT)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

def text_to_color(prompt: str):
    # Tokenisasi teks
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    
    # Generate embedding teks
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()[0]
    
    # Heuristik: Konversi embedding ke RGB (contoh sederhana)
    np.random.seed(abs(hash(prompt)) % 1000)  # Untuk konsistensi
    colors = []
    for _ in range(5):
        # Ambil 3 nilai acak dari embedding (skala 0-255)
        r = int((embeddings[0] + 1) * 127.5) % 256
        g = int((embeddings[5] + 1) * 127.5) % 256
        b = int((embeddings[10] + 1) * 127.5) % 256
        colors.append(f"#{r:02x}{g:02x}{b:02x}")
    
    return colors

# Antarmuka Gradio
with gr.Blocks(title="🎨 Text-to-Color Palette") as demo:
    gr.Markdown("# Generate Color Palette from Text")
    with gr.Row():
        prompt_input = gr.Textbox(label="Describe your colors", placeholder="e.g., 'sunset with soft pink'")
        generate_btn = gr.Button("Generate")
    with gr.Row():
        color_outputs = [gr.ColorPicker(label=f"Color {i+1}") for i in range(5)]
    
    generate_btn.click(
        fn=text_to_color,
        inputs=prompt_input,
        outputs=color_outputs
    )

demo.launch()   