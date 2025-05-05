import os
import math
from pedalboard import Pedalboard, Reverb
from pedalboard.io import AudioFile
import pickle
import numpy as np

def split_audio(input_path, output_folder, chunk_length=5):
    """Split audio file into chunks"""
    chunks = []
    with AudioFile(input_path) as af:
        sr = af.samplerate
        chunk_size = int(chunk_length * sr)
        
        for i in range(math.ceil(af.frames / chunk_size)):
            start = i * chunk_size
            af.seek(start)
            chunk = af.read(chunk_size)
            if chunk.size < sr*chunk_length:
                continue  # Skip if chunk is smaller than expected
            output_path = os.path.join(
                output_folder, 
                f"{os.path.splitext(os.path.basename(input_path))[0]}_chunk{i+1:02d}.wav"
            )
            
            with AudioFile(output_path, 'w', sr, af.num_channels) as of:
                of.write(chunk)
            chunks.append(output_path)
    return chunks

def add_reverb(input_path, output_path):
    """Add reverb to audio file"""
    board = Pedalboard([Reverb(room_size=0.7, wet_level=0.3)])
    with AudioFile(input_path) as af:
        with AudioFile(output_path, 'w', af.samplerate, af.num_channels) as of:
            chunk_size = 1024 * 1024  # Process in 1MB chunks
            for _ in range(0, af.frames, chunk_size):
                chunk = af.read(chunk_size)
                processed = board(chunk, af.samplerate)
                of.write(processed)

def process_folder(input_folder, dry_folder, wet_folder, chunk_length=5):
    """Process all audio files in folder"""
    os.makedirs(dry_folder, exist_ok=True)
    os.makedirs(wet_folder, exist_ok=True)
    
    file_mapping = {}
    
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        dry_files = split_audio(input_path, dry_folder, chunk_length)
        
        for dry_path in dry_files:
            wet_path = os.path.join(
                wet_folder, 
                os.path.basename(dry_path).replace("dry", "wet")
            )
            add_reverb(dry_path, wet_path)
            file_mapping[dry_path] = wet_path
    
    return file_mapping

if __name__ == "__main__":
    # Configuration
    input_folder = "assets/raw"
    dry_folder = "assets/dry"
    wet_folder = "assets/wet"
    
    # Run processing
    mapping = process_folder(input_folder, dry_folder, wet_folder)
    
    pickle.dump(mapping, open("assets/file_mapping.pkl", "wb"))