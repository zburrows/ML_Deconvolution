import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import random

class AudioPairDataset(Dataset):
    def __init__(self, file_pairs, transform=None, num_samples=220500):
        """
        Args:
            file_pairs (list): List of tuples (wet_path, dry_path)
            transform: Optional transform to be applied
            num_samples: Number of samples to trim/pad to
        """
        self.file_pairs = file_pairs
        self.transform = transform
        self.num_samples = num_samples

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        wet_path, dry_path = self.file_pairs[idx]
        
        def load_trim_audio(file_path):
            waveform, _ = torchaudio.load(file_path)
            return waveform[:, :self.num_samples]  # Trim to num_samples

        wet_waveform = load_trim_audio(wet_path)
        dry_waveform = load_trim_audio(dry_path)

        if self.transform:
            wet_waveform = self.transform(wet_waveform)
            dry_waveform = self.transform(dry_waveform)

        return wet_waveform, dry_waveform

class DeconvolutionModel(torch.nn.Module):
    def __init__(self):
        super(DeconvolutionModel, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

def train_model(file_pairs_dict, batch_size=16, num_epochs=100):
    # Create dataset and dataloader
    dataset = AudioPairDataset(file_pairs_dict)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeconvolutionModel().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for wet_batch, dry_batch in dataloader:
            # Move data to device and prepare dimensions
            wet_batch = wet_batch.to(device)  # Ensure shape is [batch, channels, samples]
            dry_batch = dry_batch.to(device)  # Ensure shape is [batch, channels, samples]
            
            optimizer.zero_grad()
            outputs = model(wet_batch)
            loss = criterion(outputs, dry_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.6f}")

    return model

def predict(model, audio_path, output_path=None):
    # Load and preprocess the input audio
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = torch.mean(waveform, dim=0, keepdim=True)  # Convert to mono
    waveform = waveform[:, :220500]  # Trim to match the model's input size
    waveform = waveform.unsqueeze(0).to(next(model.parameters()).device)  # Add batch dimension and move to device

    # Make prediction
    model.eval()
    with torch.no_grad():
        predicted_waveform = model(waveform)

    # Remove batch dimension and save the output
    predicted_waveform = predicted_waveform.squeeze(0).cpu()
    if output_path:
        torchaudio.save(output_path, predicted_waveform, sample_rate)

    return predicted_waveform

if __name__ == "__main__":
    # Load the file mapping dictionary
    file_pairs_dict = pickle.load(open("assets/file_mapping.pkl", "rb"))
    # Randomize the file pairs
    file_pairs_list = list(file_pairs_dict.items())
    
    # Shuffle the list of file pairs
    random.seed(12345)  # For reproducibility
    random.shuffle(file_pairs_list)
    # Train the model
    model = DeconvolutionModel()
    model.load_state_dict(torch.load("deconvolution_model.pth"))
    
    # trained_model = train_model(file_pairs_list, num_epochs=30)
    predict(model, "assets/suzanne.wav", "assets/predicted.wav")
    # Save the model
    # torch.save(trained_model.state_dict(), "deconvolution_model.pth")
    
    