import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader
import os

def trim_audio(file_path, num_samples=32000):
    waveform, _ = torchaudio.load(file_path)
    return waveform[:, :num_samples]

class AudioDataset(Dataset):
    def __init__(self, reverb_dir, dry_dir, transform=None, num_samples=32000):
        self.reverb_dir = reverb_dir
        self.dry_dir = dry_dir
        self.reverb_files = os.listdir(reverb_dir)
        self.dry_files = os.listdir(dry_dir)
        self.transform = transform
        self.num_samples = num_samples

    def __len__(self):
        return len(self.reverb_files)

    def __getitem__(self, idx):
        reverb_path = os.path.join(self.reverb_dir, self.reverb_files[idx])
        dry_path = os.path.join(self.dry_dir, self.dry_files[idx])
        reverb_waveform = trim_audio(reverb_path, self.num_samples)
        dry_waveform = trim_audio(dry_path, self.num_samples)

        if self.transform:
            reverb_waveform = self.transform(reverb_waveform)
            dry_waveform = self.transform(dry_waveform)

        return reverb_waveform, dry_waveform

reverb_dir = "assets/wet"
dry_dir = "assets/dry"
dataset = AudioDataset(reverb_dir, dry_dir)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

class DeconvolutionModel(torch.nn.Module):
    def __init__(self):
        super(DeconvolutionModel, self).__init__()
        self.conv1 = torch.nn.Conv1d(2, 16, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(16, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

model = DeconvolutionModel()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):  # Number of epochs
    for reverb_waveform, dry_waveform in dataloader:
        reverb_waveform = reverb_waveform.squeeze(1)  # Remove channel dimension
        dry_waveform = dry_waveform.squeeze(1)
        
        optimizer.zero_grad()
        output = model(reverb_waveform)
        loss = criterion(output, dry_waveform)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")