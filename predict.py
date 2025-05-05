import torch
import torchaudio
import torch.nn as nn

class AudioRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.gru(x)
        out = self.fc(out)
        return out.squeeze(-1)

def load_model(model_path, device='cpu'):
    model = AudioRNN().to(device)
    state_dict = torch.load(model_path, map_location=device)
    
    # Handle potential state dict mismatches
    if next(iter(state_dict.keys())).startswith('module.'):
        # Fix for DataParallel saved models
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # Remove 'module.'
            new_state_dict[name] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model.eval()
    return model

def preprocess_audio(waveform, sr, segment_length=22050):
    """Ensure proper shape and segmentation"""
    # Convert to mono if needed
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0)
    
    # Pad to multiple of segment length
    if (pad_len := segment_length - (len(waveform) % segment_length)) != segment_length:
        waveform = torch.nn.functional.pad(waveform, (0, pad_len))
    
    # Split into segments
    segments = waveform.unfold(0, segment_length, segment_length)
    return segments.unsqueeze(1)  # [num_segments, 1, segment_length]

def predict(model, input_path, output_path, device='cpu'):
    # Load audio
    waveform, sr = torchaudio.load(input_path)
    # Preprocess
    segments = preprocess_audio(waveform, sr)    
    # Predict
    dry_segments = []
    with torch.no_grad():
        for seg in segments:
            seg = seg.to(device)
            pred = model(seg)  # [1, segment_length]
            dry_segments.append(pred.cpu())
    
    # Combine and save
    dry_audio = torch.cat(dry_segments).flatten()
    print(dry_audio.shape)
    
    torchaudio.save(output_path, dry_audio.unsqueeze(0), sr)
    print(f"Saved output with shape: {dry_audio.unsqueeze(0).shape}")

if __name__ == "__main__":
    # Config - CHANGE THESE TO MATCH YOUR MODEL
    CONFIG = {
        'model_path': 'audio_rnn_model.pth',
        'input_audio': 'assets/miserere_short.wav',
        'output_audio': 'assets/predicted_miserere.wav',
        'segment_length': 22050  # Must match training
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    try:
        model = load_model(CONFIG['model_path'], device)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Model loading failed: {str(e)}")
        print("Check:")
        print("- Model path exists")
        print("- Model architecture matches this script")
        print("- Saved model contains proper state_dict")
        exit()

    # Run prediction
    try:
        predict(model, CONFIG['input_audio'], CONFIG['output_audio'], device)
        print("Prediction completed successfully")
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        print("Common fixes:")
        print("- Ensure input audio matches expected sample rate")
        print("- Verify segment_length matches training config")
        print("- Check audio channels (mono/stereo)")