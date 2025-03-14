import torch
import torchaudio
from pathlib import Path
import os
from tqdm import tqdm
import numpy as np
from demucs.api import Separator, save_audio
import argparse

class LongAudioSeparator:
    def __init__(self, model="mdx_extra", segment_length_seconds=15, device='cuda'):
        self.device = device
        # Initialize Separator
        self.separator = Separator(
            model=model,
            segment=segment_length_seconds,
            device=self.device,
        )
        # Model original sample rate
        self.model_sr = self.separator.samplerate
        # Target sample rate (16kHz)
        self.target_sr = 16000
        
        # Initialize resampler if needed
        if self.target_sr != self.model_sr:
            self.resampler = torchaudio.transforms.Resample(
                orig_freq=self.model_sr,
                new_freq=self.target_sr,
            ).to(self.device)
        
        print(f"Configuration:")
        print(f"- Model sample rate: {self.model_sr}Hz")
        print(f"- Output sample rate: {self.target_sr}Hz")
        print(f"- Segment length: {segment_length_seconds}s")
        print(f"- Device: {self.device}")
        print(f"- Model: {model}")
    
    def process_long_audio(self, input_path, output_dir):
        """Process long audio file"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            origin, separated = self.separator.separate_audio_file(input_path)
            
            # Only process vocals
            if 'vocals' in separated:
                vocals = separated['vocals']
                
                # Resample if needed
                if self.target_sr != self.model_sr:
                    if hasattr(vocals, 'device') and vocals.device != self.device:
                        vocals = vocals.to(self.device)
                        
                    if vocals.dim() == 2:
                        vocals = vocals.unsqueeze(0)
                    vocals = self.resampler(vocals)
                    vocals = vocals.squeeze(0)
                    if vocals.device != torch.device('cpu'):
                        vocals = vocals.to('cpu')
                
                # Save vocals file
                output_path = os.path.join(output_dir, "vocals.wav")
                save_audio(
                    vocals, 
                    output_path, 
                    samplerate=self.target_sr,
                    clip='clamp'
                )
                print(f"Saved vocals to {output_path}")
                return output_path
            else:
                raise ValueError("No vocals stem found in separation output")
        
        except Exception as e:
            print(f"Error processing {input_path}: {str(e)}")
            raise e

    def batch_process(self, input_dir, output_base_dir):
        """Batch process all audio files in directory"""
        audio_files = []
        for f in os.listdir(input_dir):
            if not f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                continue
                
            # Check if output file already exists
            output_vocal_path = os.path.join(
                output_base_dir,
                os.path.splitext(f)[0],
                "vocals.wav"
            )
            
            if not os.path.exists(output_vocal_path):
                audio_files.append(f)
        
        for audio_file in tqdm(audio_files, desc="Processing files"):
            print(f"\nProcessing {audio_file}...")
            input_path = os.path.join(input_dir, audio_file)
            output_dir = os.path.join(output_base_dir, 
                                    os.path.splitext(audio_file)[0])
            
            try:
                self.process_long_audio(input_path, output_dir)
                print(f"Successfully processed {audio_file}")
            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
                continue

    def update_parameters(self, **kwargs):
        """Update separation parameters"""
        self.separator.update_parameter(**kwargs)

def process_books(gpu_id, task_file, base_input_dir, base_output_dir):
    """Process audiobooks on specified GPU using task list"""
    # Set GPU device
    torch.cuda.set_device(gpu_id)
    print(f"Started processing on GPU {gpu_id}")
    
    # Read task list
    with open(task_file, 'r') as f:
        book_dirs = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(book_dirs)} books to process")
    
    # Initialize separator
    separator = LongAudioSeparator(
        model="mdx_extra",
        segment_length_seconds=120,
        device=f"cuda:{gpu_id}"
    )
    
    # Process each book
    for book_dir in book_dirs:
        input_dir = os.path.join(base_input_dir, book_dir)
        output_dir = os.path.join(base_output_dir, book_dir)
        print(f"Processing book: {book_dir}")
        
        try:
            separator.batch_process(input_dir, output_dir)
        except Exception as e:
            import traceback
            print(f"\n{'='*50}")
            print(f"Error processing book directory: {book_dir}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("\nDetailed traceback:")
            print(traceback.format_exc())
            print(f"{'='*50}\n")
            continue

def main():
    parser = argparse.ArgumentParser(description='Process audiobooks on specific GPU')
    parser.add_argument('--gpu', type=int, required=True, help='GPU ID to use')
    parser.add_argument('--task-file', type=str, required=True, help='Path to task file')
    parser.add_argument('--input-dir', type=str, default="data/audiobook", help='Base input directory')
    parser.add_argument('--output-dir', type=str, default="processed_data/audiobook_vocal", help='Base output directory')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs("processed_data", exist_ok=True)
    
    # Process books
    process_books(
        gpu_id=args.gpu,
        task_file=args.task_file,
        base_input_dir=args.input_dir,
        base_output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()