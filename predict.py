import torch
import torchaudio
import torch.nn as nn
import multiprocessing

# Future addition: remove pops and clicks by overlapping segments
# def create_overlapping_segments(waveform, segment_length=5000, overlap=2500):
#     """Create overlapping segments with fade-in and fade-out."""
#     # Convert to mono if needed
#     if waveform.dim() > 1:
#         waveform = waveform.mean(dim=0)
    
#     step = segment_length - overlap
#     num_segments = (len(waveform) - overlap) // step
#     segments = []

#     for i in range(num_segments):
#         start = i * step
#         end = start + segment_length
#         segment = waveform[start:end]
#         segment = torchaudio.transforms.Fade(fade_in_len=overlap, fade_out_len=overlap)(segment)
#         segments.append(segment)

#     return torch.stack(segments).unsqueeze(1)  # [num_segments, 1, segment_length]

def parallel_predict(model, segments, device='cpu'):
    """Process segments in parallel for faster operation"""
    num_workers = multiprocessing.cpu_count() # as many as available cores
    print(f"Using {num_workers} workers for parallel processing")
    # initialize the model and device for each worker
    with multiprocessing.Pool(num_workers, initializer=worker_init, initargs=(model, device)) as pool:
        dry_segments = pool.map(worker, segments) # process all segments in parallel

    return dry_segments

# Global variables for multiprocessing
_worker_model = None
_worker_device = None

def worker_init(model, device):
    """Initialize global variables for each worker."""
    global _worker_model, _worker_device
    _worker_model = model
    _worker_device = device

def worker(segment):
    """Worker function for multiprocessing."""
    global _worker_model, _worker_device
    with torch.no_grad():
        segment = segment.to(_worker_device)
        print(f"Processing segment on worker {multiprocessing.current_process().name}")
        return _worker_model(segment).cpu()

# hyperparameters copied from rnn.py
class AudioRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=512, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.gru(x)
        out = self.fc(out)
        return out.squeeze(-1)

# load from presaved model
def load_model(model_path, device='cpu'):
    model = AudioRNN().to(device)
    state_dict = torch.load(model_path, map_location=device)
    
    model.load_state_dict(state_dict)
    model.eval()
    return model

def preprocess_audio(waveform, segment_length=5000):
    # convert to mono if needed
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0)
    
    # make sure waveform is divisible by segment length
    if (pad_len := segment_length - (len(waveform) % segment_length)) != segment_length:
        waveform = torch.nn.functional.pad(waveform, (0, pad_len))
    
    # Split into segments
    segments = waveform.unfold(0, segment_length, segment_length)
    # fade for smoother transitions
    segments = [torchaudio.transforms.Fade(100, 100)(seg) for seg in segments]
    segments = torch.stack(segments)
    return segments.unsqueeze(1)

def predict(model, input_path, output_path, device='cpu'):
    waveform, sr = torchaudio.load(input_path)
    segments = preprocess_audio(waveform)    
    # predict
    dry_segments = parallel_predict(model, segments, device)
    # combine segments and flatten
    dry_audio = torch.cat(dry_segments).flatten()
    print(dry_audio.shape)
    
    torchaudio.save(output_path, dry_audio.unsqueeze(0), sr)
    print(f"Saved output {output_path} with shape: {dry_audio.unsqueeze(0).shape}")