import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class BERTColorPredictor(torch.nn.Module):
    def __init__(self, freeze_bert=False):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(768, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 15)  
        )
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.regressor(pooled_output)

def predict(prompt, model, tokenizer, device='cpu'):
    encoding = tokenizer(
        prompt,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(input_ids, attention_mask)
    
    colors = output.squeeze().cpu().numpy().reshape(-1, 3)
    
    hex_colors = []
    for color in colors:
        r, g, b = (np.clip(color, 0, 1) * 255).astype(int)
        hex_colors.append(f"#{r:02x}{g:02x}{b:02x}")
    
    return hex_colors

def visualize_color_palette(hex_colors, prompt):
    num_colors = len(hex_colors)
    fig, ax = plt.subplots(figsize=(num_colors, 2))
    
    for idx, color in enumerate(hex_colors):
        ax.add_patch(Rectangle((idx, 0), 1, 1, color=color))
        ax.text(idx + 0.5, 1.05, color, ha='center', va='bottom', fontsize=9, rotation=45)
    
    ax.set_xlim(0, num_colors)
    ax.set_ylim(0, 1.2)
    ax.axis('off')
    plt.title(f"Palet Warna untuk Prompt: '{prompt}'", fontsize=12)
    plt.show()

def main():
    MODEL_PATH = "/content/best_color_predictor.pth"  
    TOKENIZER_NAME = "bert-base-multilingual-cased"
    
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTColorPredictor()
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except RuntimeError as e:
        print(f"Error saat memuat model: {e}")
        return
    
    model.to(device)
    
    print("=== Sistem Rekomendasi Palet Warna ===")
    print("Masukkan 'exit' untuk menghentikan program.\n")
    
    while True:
        prompt = input("Masukkan prompt warna Anda: ")
        if prompt.lower() == 'exit':
            print("Program dihentikan.")
            break
        elif prompt.strip() == '':
            print("Prompt tidak boleh kosong. Silakan coba lagi.\n")
            continue
        
        hex_colors = predict(prompt, model, tokenizer, device)
        
        print(f"\nPrompt: {prompt}")
        print("Warna:", hex_colors)
        
        visualize_color_palette(hex_colors, prompt)
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()
