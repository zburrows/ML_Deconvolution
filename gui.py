import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import torch
from predict import load_model, predict
from pydub import AudioSegment
from pydub.playback import play
import os
TMP_PATH = "assets/tmp.wav"

class AudioFileApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Deconvolution Demo")
        self.root.geometry("400x300")
        
        self.label = tk.Label(root, text="No file selected", pady=20)
        self.label.pack()
        
        # open file dialog
        self.button = tk.Button(
            root, 
            text="Select Audio File", 
            command=self.open_file_dialog,
            height=2,
            width=20
        )
        self.button.pack(pady=50)
        
        # supported file types
        self.file_types = [
            ("Audio files", "*.wav *.aiff *.flac *.mp3"),
            ("All files", "*.*")
        ]

    def open_file_dialog(self):
        try:
            file_path = filedialog.askopenfilename( # choose input audio file
                title="Select an audio file",
                initialdir=str(Path.home() / "Documents"),
                filetypes=self.file_types
            )
            
            if file_path: # if file selected
                filename = Path(file_path).name
                self.label.config(text=f"Selected: {filename}")
                print(f"File selected: {file_path}")
                process_audio(file_path, TMP_PATH)
                # Create a frame for additional options
                options_frame = tk.Frame(self.root)
                options_frame.pack(pady=20)

                # play button
                play_button = tk.Button(
                    options_frame,
                    text="Play Processed Audio",
                    command=lambda: self.play_audio(TMP_PATH), # if clicked open external player
                    height=2,
                    width=20
                )
                play_button.grid(row=0, column=0, padx=10)

                # Save button
                save_button = tk.Button(
                    options_frame,
                    text="Save Processed Audio",
                    command=lambda: self.save_audio(Path(TMP_PATH)), # if clicked open save dialog
                    height=2,
                    width=20
                )
                save_button.grid(row=0, column=1, padx=10)
            else:
                self.label.config(text="No file selected")
        except Exception as e:
            self.label.config(text=f"Error: {str(e)}")
            print(f"Error: {e}")


    def play_audio(self, file_path):
        try:
            audio = AudioSegment.from_file(file_path)
            audio.export("temp_output.wav", format="wav")
            os.system("start temp_output.wav")
            print(f"Playing audio: {file_path}")
        except Exception as e:
            print(f"Error playing audio: {e}")

    def save_audio(self, file_path):
        try:
            save_path = filedialog.asksaveasfilename(
                title="Save Processed Audio",
                defaultextension=".wav",
                filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
            )
            if save_path:
                Path(file_path).rename(save_path)
                print(f"Audio saved to: {save_path}")
        except Exception as e:
            print(f"Error saving audio: {e}")


def process_audio(file_path, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # use appropriate device
    print(f"Using device: {device}")
    
    model = load_model("audio_rnn_model.pth", device) # run from pretrained model

    predict(model, file_path, output_path, device)
    print("Prediction completed successfully")

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioFileApp(root)
    root.mainloop()