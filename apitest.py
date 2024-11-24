import requests
import base64
import wave
import numpy as np

def test_voice_chat():
    base_url = "http://localhost:8000"

    print("Setting reference voice...")
    # First set the reference voice
    with open('untitled.wav', 'rb') as f:
        files = {
            'audio_file': ('untitled.wav', f, 'audio/wav')
        }
        data = {
            'reference_text': ''  # Optional reference text
        }
        response = requests.post(
            f'{base_url}/set-reference-voice',
            files=files,
            data=data
        )
    
    try:
        print("Set reference voice response:", response.json())
    except requests.exceptions.JSONDecodeError:
        print("Error response:", response.text)
        return

    print("\nTesting text chat...")
    # Test text chat
    chat_data = {"text": "Hello, how are you today?"}
    response = requests.post(
        f'{base_url}/chat/text',
        json=chat_data
    )
    
    try:
        response_data = response.json()
        if 'error' in response_data:
            print("Error:", response_data['error'])
            if 'traceback' in response_data:
                print("Traceback:", response_data['traceback'])
            return
            
        print("Chat response text:", response_data['response_text'])
        
        # Save the audio response
        if 'audio_response' in response_data:
            audio_bytes = base64.b64decode(response_data['audio_response'])
            sample_rate = response_data.get('sample_rate', 24000)  # Default to 24000 if not provided
            
            with open('response.wav', 'wb') as f:
                f.write(audio_bytes)
            print(f"Saved audio response to response.wav (sample rate: {sample_rate}Hz)")
        else:
            print("No audio response in data")
            
    except requests.exceptions.JSONDecodeError:
        print("Error in chat response:", response.text)
    except Exception as e:
        print(f"Error processing response: {str(e)}")
        print("Response content:", response.content[:200])  # Print first 200 bytes of response

if __name__ == "__main__":
    test_voice_chat()