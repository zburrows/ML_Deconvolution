import os
from pedalboard import Pedalboard, Reverb
from pedalboard.io import AudioFile
import pickle

def add_reverb(input_path, output_path):
    """Add reverb to audio file"""
    board = Pedalboard([Reverb(room_size=0.9, wet_level=0.3)])
    with AudioFile(input_path) as af:
        with AudioFile(output_path, 'w', af.samplerate, af.num_channels) as of:
            chunk_size = 1024 * 1024
            for _ in range(0, af.frames, chunk_size):
                chunk = af.read(chunk_size)
                processed = board(chunk, af.samplerate)
                of.write(processed)

def process_folder(input_folder, wet_folder):
    """Process all audio files in folder"""
    
    file_mapping = {}
    
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        wet_path = os.path.join(
            wet_folder, 
            os.path.basename(input_path).replace("dry", "wet")
        )
        add_reverb(input_path, wet_path)
        file_mapping[input_path] = wet_path
    
    return file_mapping

if __name__ == "__main__":
    input_folder = "assets/dry"
    wet_folder = "assets/wet"
    
    # add reverb and dump to wet folder
    mapping = process_folder(input_folder, wet_folder)
    # save mapping to pickle file
    pickle.dump(mapping, open("assets/file_mapping.pkl", "wb"))