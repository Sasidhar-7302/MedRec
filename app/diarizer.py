
import whisperx
import torch
import gc
import os

class Diarizer:
    def __init__(self, device="cuda", compute_type="float16", model_name="large-v3"):
        self.device = device
        self.compute_type = compute_type
        self.model = None
        self.align_model = None
        self.diarize_model = None
        self.model_name = model_name
        print(f"Initializing Diarizer on {self.device} with {self.compute_type}...")

    def load_resources(self):
        """Loads models if they aren't already loaded to save VRAM when not in use."""
        if self.model is None:
            print(f"Loading WhisperX model: {self.model_name}...")
            self.model = whisperx.load_model(
                self.model_name, 
                self.device, 
                compute_type=self.compute_type
            )
        
    def process_audio(self, audio_file, num_speakers=None, min_speakers=None, max_speakers=None, initial_prompt=None):
        """
        Full pipeline: Transcribe -> Align -> Diarize
        """
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        self.load_resources()
        
        # 1. Transcribe
        print(f"Transcribing (batch_size=16, device={self.device})...")
        audio = whisperx.load_audio(audio_file)
        result = self.model.transcribe(audio, batch_size=16)
        
        # 2. Align
        print("Aligning...")
        if self.align_model is None:
             self.align_model, self.metadata = whisperx.load_align_model(
                 language_code=result["language"], 
                 device=self.device
             )
        
        result = whisperx.align(
            result["segments"], 
            self.align_model, 
            self.metadata, 
            audio, 
            self.device, 
            return_char_alignments=False
        )
        
        print(f"Diarizing...")
        token = os.getenv("HF_TOKEN")
        if token:
            print(f"HF_TOKEN found: {token[:4]}...{token[-4:]}")
        else:
            print("HF_TOKEN not found in environment variables.")
        
        if self.diarize_model is None:
            # Note: You need a HF token for this usually. 
            # We assume the user has logged in or provided it in env.
            # If not, this might fail.
            # Adjusted import to fix AttributeError
            from whisperx.diarize import DiarizationPipeline
            self.diarize_model = DiarizationPipeline(
                use_auth_token=os.getenv("HF_TOKEN"),
                device=self.device
            )
            
        diarize_segments = self.diarize_model(
            audio, 
            min_speakers=min_speakers, 
            max_speakers=max_speakers,
            num_speakers=num_speakers
        )
        
        # 4. Assign Speakers to Words
        result = whisperx.assign_word_speakers(diarize_segments, result)
        
        # 5. Format Output
        final_segments = []
        for segment in result["segments"]:
            final_segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip(),
                "speaker": segment.get("speaker", "UNKNOWN")
            })
            
        return final_segments

    def cleanup(self):
        """Free VRAM"""
        del self.model
        del self.align_model
        del self.diarize_model
        self.model = None
        self.align_model = None
        self.diarize_model = None
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Test stub
    import sys
    if len(sys.argv) > 1:
        diarizer = Diarizer()
        res = diarizer.process_audio(sys.argv[1])
        for seg in res:
            print(f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['speaker']}: {seg['text']}")
