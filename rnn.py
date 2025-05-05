import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import random
import numpy as np
from tqdm import tqdm
import pickle

# Configuration
config = {
    'sample_rate': 44100,
    'segment_length': 44100,  # 1 second of audio
    'batch_size': 32,
    'hidden_size': 256,
    'num_layers': 3,
    'learning_rate': 0.0001,
    'num_epochs': 100,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

class AudioPairDataset(Dataset):
    def __init__(self, file_pairs, segment_length):
        self.file_pairs = file_pairs
        self.segment_length = segment_length
        
    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        wet_path, dry_path = self.file_pairs[idx]
        
        # Load and preprocess wet (input) and dry (target) audio
        wet, _ = torchaudio.load(wet_path)
        dry, _ = torchaudio.load(dry_path)
        
        # Convert to mono if needed
        if wet.shape[0] > 1:
            wet = wet.mean(dim=0, keepdim=True)
        if dry.shape[0] > 1:
            dry = dry.mean(dim=0, keepdim=True)
            
        # Randomly select a segment
        if wet.shape[1] > self.segment_length:
            start = torch.randint(0, wet.shape[1] - self.segment_length, (1,))
            wet = wet[:, start:start+self.segment_length]
            dry = dry[:, start:start+self.segment_length]
        else:
            # Pad if shorter than segment length
            pad = self.segment_length - wet.shape[1]
            wet = torch.nn.functional.pad(wet, (0, pad))
            dry = torch.nn.functional.pad(dry, (0, pad))
            
        return wet.squeeze(0), dry.squeeze(0)  # Remove channel dim

class AudioRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=1):
        super(AudioRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Using GRU for better performance with audio
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len)
        x = x.unsqueeze(-1)  # (batch, seq_len, 1)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass
        out, _ = self.gru(x, h0)
        out = self.fc(out)
        return out.squeeze(-1)  # (batch, seq_len)

def train_model(file_pairs, config):
    # Create dataset and dataloader
    dataset = AudioPairDataset(file_pairs, config['segment_length'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Initialize model
    model = AudioRNN(
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers']
    ).to(config['device'])
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    for epoch in range(config['num_epochs']):
        model.train()
        running_loss = 0.0
        
        for wet, dry in dataloader:
            wet = wet.to(config['device'])
            dry = dry.to(config['device'])
            
            # Forward pass
            outputs = model(wet)
            loss = criterion(outputs, dry)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.6f}")
    
    return model

def predict_dry_signal(model, wet_audio_path, output_path=None, config=config):
    """Predict dry signal from wet audio using trained model"""
    model.eval()
    
    # Load and preprocess wet audio
    wet, sr = torchaudio.load(wet_audio_path)
    if wet.shape[0] > 1:
        wet = wet.mean(dim=0, keepdim=True)
    
    # Pad to multiple of segment length
    pad_len = config['segment_length'] - (wet.shape[1] % config['segment_length'])
    wet = torch.nn.functional.pad(wet, (0, pad_len))
    
    # Process in segments
    dry_pred = []
    for i in range(0, wet.shape[1], config['segment_length']):
        segment = wet[:, i:i+config['segment_length']].to(config['device'])
        with torch.no_grad():
            pred_segment = model(segment.squeeze(0)).cpu()
        dry_pred.append(pred_segment)
    
    # Combine segments and remove padding
    dry_pred = torch.cat(dry_pred, dim=1)[:, :wet.shape[1]]
    
    # Save if output path provided
    if output_path:
        torchaudio.save(output_path, dry_pred.unsqueeze(0), sr)
    
    return dry_pred

if __name__ == "__main__":
    # Example file pairs (wet -> dry)
    file_pairs_dict = pickle.load(open("assets/file_mapping.pkl", "rb"))
    # Randomize the file pairs
    file_pairs_list = list(file_pairs_dict.items())
    file_pairs_list = [('assets/dry/swing_dry.wav', 'assets/wet/swing_wet.wav'), ('assets/dry/hymn_dry.wav', 'assets/wet/hymn_wet.wav'), ('assets/dry/flamenco_dry.wav', 'assets/wet/flamenco_wet.wav'), ('assets/dry/brahms_dry.wav', 'assets/wet/brahms_wet.wav'), ('assets/dry/mozart_dry.wav', 'assets/wet/mozart_wet.wav')]
    # Shuffle the list of file pairs
    random.seed(12345)  # For reproducibility
    random.shuffle(file_pairs_list)
    print(file_pairs_list)
    # Train the model
    print("Training RNN model...")
    model = train_model(file_pairs_list, config)
    
    # Save the trained model
    torch.save(model.state_dict(), "audio_rnn_model.pth")
    print("Model saved to audio_rnn_model.pth")