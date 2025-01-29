import json
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class ColorDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        with open(data_path) as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        for item in self.data:
            colors = []
            for hex_color in item["colors"]:
                hex = hex_color.lstrip('#')
                rgb = tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))
                colors.append([x/255.0 for x in rgb])
            item["colors"] = colors
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["prompt"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        colors = torch.tensor(item["colors"], dtype=torch.float32)
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "colors": colors.view(-1)  
        }
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
    
def train_model():
    BATCH_SIZE = 4
    EPOCHS = 50
    LEARNING_RATE = 2e-5
    MODEL_SAVE_PATH = "best_color_predictor.pth"  
    DATA_PATH = "/content/data.json"  
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    dataset = ColorDataset(DATA_PATH, tokenizer)
    
    train_indices, val_indices = train_test_split(
        list(range(len(dataset))), test_size=0.2, random_state=42
    )
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)
    
    model = BERTColorPredictor(freeze_bert=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    best_val_loss = float('inf')  
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["colors"].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["colors"].to(device)
                
                outputs = model(input_ids, attention_mask)
                val_loss += criterion(outputs, targets).item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}\n")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Model terbaik disimpan dengan Val Loss: {best_val_loss:.4f}\n")
    
    print("Training selesai! Model terbaik disimpan sebagai", MODEL_SAVE_PATH)
def predict(prompt, model_path="best_color_predictor.pth"):
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = BERTColorPredictor()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    encoding = tokenizer(
        prompt,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        output = model(encoding["input_ids"], encoding["attention_mask"])
    
    colors = output.squeeze().numpy().reshape(-1, 3)
    hex_colors = []
    for color in colors:
        r, g, b = (np.clip(color, 0, 1) * 255).astype(int)
        hex_colors.append(f"#{r:02x}{g:02x}{b:02x}")
    
    return hex_colors

if __name__ == "__main__":
    train_model()
    
    test_prompt = "nuansa biru langit dengan gradasi putih awan"
    predicted_colors = predict(test_prompt)
    
    print("\nContoh Prediksi:")
    print(f"Prompt: {test_prompt}")
    print("Warna:", predicted_colors)