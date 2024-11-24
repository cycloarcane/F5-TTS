from f5_tts.api import F5TTS
from f5_tts.infer.utils_infer import transcribe  # Using F5-TTS's built-in transcription
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import soundfile as sf
import sounddevice as sd
import numpy as np
from scipy import signal
import tempfile
import os
import wave
import threading
import queue
import time

class VoiceChat:
    def __init__(self, character_prompt="You are not an AI assistant, you are whoever the user says you are. Keep responses concise since they will be spoken."):
        """Initialize the voice chat system"""
        print("Initializing voice chat system...")
        
        # Initialize F5-TTS
        self.tts = F5TTS(
            model_type="F5-TTS",
            vocoder_name="vocos",
            device=None  # will auto-select cuda/mps/cpu
        )
        
        # Initialize LLM
        print("Loading chat model...")
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        self.chat_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize conversation state
        self.conversation = [{
            "role": "system",
            "content": character_prompt
        }]
        
        # Store reference voice information
        self.reference_audio = None
        self.reference_text = None
        
        # Audio settings
        self.device_info = sd.query_devices(sd.default.device[1])
        self.supported_samplerate = int(self.device_info['default_samplerate'])
        self.channels = 1  # Mono recording
        self.record_sample_rate = 16000  # Standard rate for speech recognition
        
        print(f"Audio device: {self.device_info['name']}")
        print(f"Supported sample rate: {self.supported_samplerate}")
        
        print("Voice chat system initialized!")

        # Try to find a working output device
        devices = sd.query_devices()
        print("\nAvailable audio devices:")
        for i, dev in enumerate(devices):
            print(f"{i}: {dev['name']} (max outputs: {dev['max_output_channels']})")

        # Let user select output device
        device_id = input("\nSelect output device number (press Enter for default): ").strip()
        if device_id:
            sd.default.device[1] = int(device_id)
    
    def record_audio(self, duration=None):
        """Record audio from microphone"""
        print("Recording... Press Ctrl+C to stop.")
        
        # Create a queue to store audio data
        q = queue.Queue()
        
        # Callback function to store audio data
        def callback(indata, frames, time, status):
            if status:
                print(status)
            q.put(indata.copy())
        
        # Create a temporary file to store the recording
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        
        try:
            # Open an input stream
            with sd.InputStream(samplerate=self.record_sample_rate,
                              channels=self.channels,
                              callback=callback):
                
                # Open a WAV file to write the recorded audio
                with wave.open(temp_file.name, 'wb') as file:
                    file.setnchannels(self.channels)
                    file.setsampwidth(2)  # 16-bit audio
                    file.setframerate(self.record_sample_rate)
                    
                    print("Recording started (Press Ctrl+C to stop)...")
                    
                    # Record until duration or Ctrl+C
                    start_time = time.time()
                    while True:
                        if duration and (time.time() - start_time) > duration:
                            break
                        file.writeframes(q.get().tobytes())
                        
        except KeyboardInterrupt:
            print("\nRecording stopped.")
        
        return temp_file.name
    
    def set_reference_voice(self, reference_audio_path, reference_text=""):
        """Set the reference voice to be used for responses"""
        self.reference_audio = reference_audio_path
        self.reference_text = reference_text
        print(f"Reference voice set from: {reference_audio_path}")
    
    def generate_response(self, messages):
        """Generate response using the LLM"""
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.chat_model.device)
        generated_ids = self.chat_model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
        )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def play_audio(self, wav_data, sample_rate):
            """Play audio using sounddevice with improved error handling"""
            try:
                print(f"\nPlayback debug info:")
                print(f"Input sample rate: {sample_rate}")
                print(f"Output sample rate: {self.supported_samplerate}")
                print(f"Audio data shape: {wav_data.shape}")
                print(f"Audio data type: {wav_data.dtype}")
                print(f"Audio data range: {wav_data.min():.3f} to {wav_data.max():.3f}")
                
                # Convert to float32 if not already
                wav_data = wav_data.astype(np.float32)
                
                # Resample if necessary
                if sample_rate != self.supported_samplerate:
                    print(f"Resampling from {sample_rate} to {self.supported_samplerate}")
                    number_of_samples = round(len(wav_data) * float(self.supported_samplerate) / sample_rate)
                    wav_data = signal.resample(wav_data, number_of_samples)
                
                # Ensure audio is in the correct range (-1 to 1)
                if wav_data.max() > 1 or wav_data.min() < -1:
                    print("Normalizing audio levels...")
                    wav_data = wav_data / max(abs(wav_data.max()), abs(wav_data.min()))
                
                print("Starting playback...")
                
                try:
                    # First try default device
                    sd.play(wav_data, self.supported_samplerate)
                    sd.wait()
                except Exception as e:
                    print(f"Failed to play on default device: {e}")
                    print("Trying alternative playback method...")
                    
                    # Try to find a working output device
                    devices = sd.query_devices()
                    for i, device in enumerate(devices):
                        if device['max_output_channels'] > 0:
                            try:
                                print(f"Trying device {i}: {device['name']}")
                                sd.play(wav_data, self.supported_samplerate, device=i)
                                sd.wait()
                                print(f"Successfully played audio on device: {device['name']}")
                                break
                            except Exception as dev_e:
                                print(f"Failed on device {i}: {dev_e}")
                                continue
                
                print("Playback completed!")
                
            except Exception as e:
                print(f"Error playing audio: {e}")
                print("Saving audio to file instead...")
                output_path = "latest_response.wav"
                sf.write(output_path, wav_data, self.supported_samplerate)
                print(f"Audio saved to: {output_path}")
                
                # Print available audio devices for debugging
                print("\nAvailable audio devices:")
                print(sd.query_devices())
    
    def process_voice_input(self):
        """Record and process voice input"""
        # Record audio
        audio_file = self.record_audio()
        
        # Transcribe audio
        print("Transcribing your message...")
        message = transcribe(audio_file)
        print(f"You said: {message}")
        
        # Clean up temporary file
        os.unlink(audio_file)
        
        return message
    
    def process_message(self, message, output_path=None, play_audio=True):
        """Process a message and generate an audio response"""
        if self.reference_audio is None:
            raise ValueError("Reference voice not set! Call set_reference_voice first.")
        
        # Print the current conversation for context
        print("\nConversation history:")
        for msg in self.conversation:
            if msg["role"] != "system":
                print(f"{msg['role'].title()}: {msg['content']}")
            
        # Add user message to conversation
        self.conversation.append({
            "role": "user",
            "content": message
        })
        print(f"\nUser: {message}")
        
        # Generate AI response
        response = self.generate_response(self.conversation)
        
        # Add AI response to conversation
        self.conversation.append({
            "role": "assistant",
            "content": response
        })
        
        print(f"Assistant: {response}")
        
        # Generate speech for the response
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            output_path = temp_file.name
            temp_file.close()
        
        wav, sr, spect = self.tts.infer(
            ref_file=self.reference_audio,
            ref_text=self.reference_text,
            gen_text=response,
            file_wave=output_path,
            remove_silence=True,
            speed=1.0
        )
        
        # Play the audio if requested
        if play_audio:
            print("Playing response...")
            self.play_audio(wav, sr)
        
        return {
            "response_text": response,
            "audio_path": output_path,
            "sample_rate": sr,
            "waveform": wav
        }
    
    def reset_conversation(self, new_system_prompt=None):
        """Reset the conversation history"""
        if new_system_prompt:
            self.conversation = [{
                "role": "system",
                "content": new_system_prompt
            }]
        else:
            self.conversation = [self.conversation[0]]
        print("Conversation reset!")

def voice_chat():
    """Interactive voice chat function"""
    print("\n=== F5-TTS Voice Chat ===")
    
    # Initialize the chat system
    chat = VoiceChat(
        character_prompt="You are a friendly assistant named Alex. Keep your responses short and conversational."
    )
    
    # Set reference voice
    reference_file = input("\nEnter path to reference voice file (default: untitled.wav): ").strip()
    if not reference_file:
        reference_file = "untitled.wav"
    
    chat.set_reference_voice(reference_file)
    
    print("\nChat started! Press Ctrl+C during recording to stop and send message.")
    print("Type 'quit' to exit, 'reset' to start over, or press Enter to start recording.")
    
    while True:
        try:
            user_input = input("\nPress Enter to start recording (or type 'quit'/'reset'): ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'reset':
                chat.reset_conversation()
                print("Conversation reset!")
                continue
            elif user_input:  # If they typed something else, use it as text input
                message = user_input
            else:  # Empty input (just Enter) means start recording
                message = chat.process_voice_input()
            
            if message:  # If we got a message (either typed or recorded)
                chat.process_message(message)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    voice_chat()