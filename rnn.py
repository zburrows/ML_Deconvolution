import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import random

# hyperparameters
config = {
    'sample_rate': 44100,
    'segment_length': 44100,
    'batch_size': 32,
    'hidden_size': 256,
    'num_layers': 3,
    'learning_rate': 0.0001,
    'num_epochs': 100,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# read in a pair of audio files
class AudioPairDataset(Dataset):
    def __init__(self, file_pairs, segment_length):
        self.file_pairs = file_pairs
        self.segment_length = segment_length
        
    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        wet_path, dry_path = self.file_pairs[idx]
        
        # load wet and dry files
        wet, _ = torchaudio.load(wet_path)
        dry, _ = torchaudio.load(dry_path)
        
        # mono conversion
        if wet.shape[0] > 1: # more than 1 channel
            wet = wet.mean(dim=0, keepdim=True)
        if dry.shape[0] > 1:
            dry = dry.mean(dim=0, keepdim=True)
            
        # choose a random segment
        if wet.shape[1] > self.segment_length:
            start = torch.randint(0, wet.shape[1] - self.segment_length, (1,)) # random start index
            wet = wet[:, start:start+self.segment_length] # retrieve segment
            dry = dry[:, start:start+self.segment_length]
        else:
            # Pad if shorter than segment length
            pad = self.segment_length - wet.shape[1]
            wet = torch.nn.functional.pad(wet, (0, pad))
            dry = torch.nn.functional.pad(dry, (0, pad))
            
        return wet.squeeze(0), dry.squeeze(0) # return audio data only

class AudioRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=1):
        super(AudioRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len)
        x = x.unsqueeze(-1)  # (batch, seq_len, 1)
        
        # hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.gru(x, h0)
        out = self.fc(out)
        return out.squeeze(-1)  # (batch, seq_len)

def train_model(file_pairs, config):
    dataset = AudioPairDataset(file_pairs, config['segment_length'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # create model with hyperparameters
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
            
            outputs = model(wet)
            loss = criterion(outputs, dry)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.6f}")
    
    return model

if __name__ == "__main__":
    # locate files for training
    file_pairs_list = [('assets/dry/swing_dry.wav', 'assets/wet/swing_wet.wav'), ('assets/dry/hymn_dry.wav', 'assets/wet/hymn_wet.wav'), ('assets/dry/flamenco_dry.wav', 'assets/wet/flamenco_wet.wav'), ('assets/dry/brahms_dry.wav', 'assets/wet/brahms_wet.wav'), ('assets/dry/mozart_dry.wav', 'assets/wet/mozart_wet.wav')]
    # shuffle the list of file pairs
    random.seed(12345)  # for reproducibility
    random.shuffle(file_pairs_list)
    
    # train the model
    print("Training RNN...")
    model = train_model(file_pairs_list, config)
    
    # save the trained model
    torch.save(model.state_dict(), "audio_rnn_model.pth")
    print("Model saved to audio_rnn_model.pth")