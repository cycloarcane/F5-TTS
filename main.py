from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import base64
import torch
import torchaudio
from pathlib import Path
import tempfile
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from cached_path import cached_path
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
)

from pydantic import BaseModel


class TextRequest(BaseModel):
    text: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize models when the server starts
    print("Initializing models...")
    global chat_model, tokenizer, asr_pipeline, tts_model, vocoder
    
    if chat_model is None:
        print("Loading chat model...")
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        chat_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype="auto", 
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if asr_pipeline is None:
        print("Loading ASR pipeline...")
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            "openai/whisper-large-v3-turbo",
            torch_dtype=torch.float16,
            device="cuda"
        )

    if tts_model is None:
        print("Loading TTS model...")
        model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
        ckpt_file = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"))
        tts_model = load_model(DiT, model_cfg, ckpt_file)

    if vocoder is None:
        print("Loading vocoder...")
        vocoder = load_vocoder()
    
    yield  # Server is running
    
    # Cleanup (if needed) when server shuts down
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)



# Global model states
chat_model = None 
tokenizer = None
asr_pipeline = None
tts_model = None
vocoder = None

# Store reference voice info
reference_data = {
    "audio": None,
    "text": None
}

def initialize_models():
    global chat_model, tokenizer, asr_pipeline, tts_model, vocoder
    
    if chat_model is None:
        print("Loading chat model...")
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        chat_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype="auto", 
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if asr_pipeline is None:
        print("Loading ASR pipeline...")
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            "openai/whisper-large-v3-turbo",
            torch_dtype=torch.float16,
            device="cuda"
        )

    if tts_model is None:
        print("Loading TTS model...")
        model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
        ckpt_file = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"))
        tts_model = load_model(DiT, model_cfg, ckpt_file)

    if vocoder is None:
        print("Loading vocoder...")
        vocoder = load_vocoder()



@app.post("/set-reference-voice")
async def set_reference_voice(
    audio_file: UploadFile = File(...),
    reference_text: str = Form(None)  # Changed to Form parameter
):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(await audio_file.read())
            temp_path = temp_file.name

        # Process reference audio
        processed_audio, processed_text = preprocess_ref_audio_text(
            temp_path, 
            reference_text if reference_text else "",  # Provide empty string if None
            show_info=print
        )
        
        reference_data["audio"] = processed_audio
        reference_data["text"] = processed_text
        
        os.unlink(temp_path)  # Clean up temp file
        
        return JSONResponse(content={
            "message": "Reference voice set successfully",
            "transcribed_text": processed_text
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/chat/audio")
async def chat_audio(audio_file: UploadFile = File(...)):
    if reference_data["audio"] is None:
        raise HTTPException(400, "Reference voice not set")
        
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(await audio_file.read())
            temp_path = temp_file.name

        # 1. Speech to text
        user_text = asr_pipeline(temp_path)["text"].strip()
        
        # 2. Generate chat response
        messages = [
            {
                "role": "system",
                "content": "You are having a natural conversation. Keep responses concise since they will be spoken."
            },
            {
                "role": "user", 
                "content": user_text
            }
        ]
        
        chat_response = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer([chat_response], return_tensors="pt").to(chat_model.device)
        generated_ids = chat_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
        )
        
        response_text = tokenizer.batch_decode(
            [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)],
            skip_special_tokens=True
        )[0]

        # 3. Text to speech
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_file:
            audio_wave, sample_rate, _ = infer_process(
                reference_data["audio"],
                reference_data["text"], 
                response_text,
                tts_model,
                vocoder,
                show_info=print
            )
            
            torchaudio.save(out_file.name, 
                           torch.tensor(audio_wave).unsqueeze(0), 
                           sample_rate)
            
            response = FileResponse(
                out_file.name,
                media_type="audio/wav",
                filename="response.wav"
            )
            
        os.unlink(temp_path)  # Clean up input temp file
        
        return JSONResponse(content={
            "user_text": user_text,
            "response_text": response_text,
            "audio_response": response
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/chat/text") 
async def chat_text(request: TextRequest):
    if reference_data["audio"] is None:
        raise HTTPException(400, "Reference voice not set")
        
    try:
        # Generate chat response
        messages = [
            {
                "role": "system",
                "content": "You are having a natural conversation. Keep responses concise since they will be spoken."
            },
            {
                "role": "user",
                "content": request.text
            }
        ]
        
        chat_response = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer([chat_response], return_tensors="pt").to(chat_model.device)
        generated_ids = chat_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
        )
        
        response_text = tokenizer.batch_decode(
            [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)],
            skip_special_tokens=True
        )[0]

        # Text to speech
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_file:
            audio_wave, sample_rate, _ = infer_process(
                reference_data["audio"],
                reference_data["text"],
                response_text, 
                tts_model,
                vocoder,
                show_info=print
            )
            
            torchaudio.save(out_file.name,
                           torch.tensor(audio_wave).unsqueeze(0),
                           sample_rate)
            
            # Read the audio file and convert to base64
            with open(out_file.name, 'rb') as audio_file:
                audio_bytes = base64.b64encode(audio_file.read()).decode('utf-8')
            
            # Clean up temp file
            os.unlink(out_file.name)
            
        return JSONResponse(content={
            "response_text": response_text,
            "audio_response": audio_bytes,
            "sample_rate": sample_rate
        })
        
    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )
    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)