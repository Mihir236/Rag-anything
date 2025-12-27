import asyncio
import os
import shutil
import argparse
from dotenv import load_dotenv

# Load environment variables immediately
load_dotenv()

# Map Nvidia/LightRAG keys to OpenAI format if needed
if os.getenv("LLM_BINDING_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("LLM_BINDING_API_KEY")
if os.getenv("LLM_BINDING_HOST"):
    os.environ["OPENAI_BASE_URL"] = os.getenv("LLM_BINDING_HOST")

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

async def main():
    parser = argparse.ArgumentParser(description="Run RAGAnything on a document")
    parser.add_argument("file_path", nargs="?", default="/Users/mihirmodi/Desktop/Student List.xlsx", help="Path to the file to process")
    args = parser.parse_args()

    # Optional: Clear old storage to force fresh processing (recommended first time)
    if os.path.exists("./rag_storage"):
        print("Clearing old rag_storage for fresh start...")
        shutil.rmtree("./rag_storage")

    # Config - using docling parser for Excel support
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="mineru",                   # Switched to mineru for Image/PDF support
        # parser="docling",                  # Keep docling as option for simple Excel
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    # LLM: Use powerful model from env (defaults to nemotron-70b)
    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            model=os.environ.get("LLM_MODEL", "meta/llama-3.1-8b-instruct"),
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            base_url=os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY"),
            **kwargs,
        )

    # Vision model: Use powerful VLM from env (defaults to neva-22b)
    def vision_model_func(prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs):
        # Fallback to text-only for non-image cases
        return openai_complete_if_cache(
            model=os.environ.get("VISION_MODEL", "nvidia/neva-22b"),
            messages=messages or [{"role": "user", "content": prompt}],
            base_url=os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY"),
            **kwargs,
        )

    # Embeddings: Use a model known to work well with NVIDIA's endpoint
    embedding_func = EmbeddingFunc(
        embedding_dim=4096,  # Matches nv-embed-v1
        max_token_size=2048,
        func=lambda texts, max_token_size=None: openai_embed.func(
            texts,
            model=os.environ.get("EMBEDDING_MODEL", "nvidia/nv-embed-v1"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY"),
            max_token_size=max_token_size,
        ),
    )

    # Initialize RAGAnything
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )

    # Your Excel file path
    file_path = args.file_path
    
    # Handle Video Files (simple frame extraction)
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # Batch Processing (Folder)
    if os.path.isdir(file_path):
        print(f"Detected directory: {file_path}")
        print("Starting batch processing...")
        await rag.process_folder_complete(
            folder_path=file_path,
            output_dir="./output",
            file_extensions=[".pdf", ".docx", ".pptx", ".xlsx", ".jpg", ".png", ".txt", ".md"],
            recursive=True,
            max_workers=4
        )
    
    else:
        # Check file extension
        _, ext = os.path.splitext(file_path)

        if ext.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            print(f"Detected video file: {file_path}")
            print("Extracting key frames for processing...")
            
            try:
                import cv2
            except ImportError:
                print("Error: 'opencv-python' is required for video processing.")
                print("Please run: pip install opencv-python")
                return

            # Create frames directory
            frames_dir = os.path.join(os.path.dirname(file_path), f"{os.path.basename(file_path)}_frames")
            if os.path.exists(frames_dir):
                shutil.rmtree(frames_dir)
            os.makedirs(frames_dir)

            cap = cv2.VideoCapture(file_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_interval = int(fps * 2)  # Extract 1 keyframe every 2 seconds
            
            count = 0
            saved_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if count % frame_interval == 0:
                    frame_path = os.path.join(frames_dir, f"frame_{saved_count:04d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    saved_count += 1
                count += 1
            
            cap.release()
            print(f"Extracted {saved_count} frames to {frames_dir}")
            print("Processing frames as a document folder...")
            
            # Process the folder of frames instead of the video file
            await rag.process_document_complete(
                file_path=frames_dir,  # Pass the folder!
                output_dir="./output",
                parse_method="auto"
            )
        
        # Handle Audio Files (Transcription)
        elif ext.lower() in ['.mp3', '.wav', '.m4a']:
            print(f"Detected audio file: {file_path}")
            print("Loading Whisper model for transcription...")
            
            try:
                import whisper
            except ImportError:
                print("Error: 'openai-whisper' is required for audio processing.")
                print("Please run: pip install openai-whisper")
                return

            # Transcribe
            model = whisper.load_model("base") # Use base model for speed
            print("Transcribing audio (this may take a while)...")
            result = model.transcribe(file_path)
            transcription = result["text"]
            
            print("Transcription complete!")
            
            # Save to text file
            txt_path = os.path.join(os.path.dirname(file_path), f"{os.path.basename(file_path)}_transcription.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"Transcription of {os.path.basename(file_path)}:\n\n")
                f.write(transcription)
            
            print(f"Saved transcription to {txt_path}")
            print("Processing transcribed text...")
            
            await rag.process_document_complete(
                file_path=txt_path,
                output_dir="./output",
                parse_method="auto"
            )

        else:
            # Standard processing for Excel, Images, PDFs
            print(f"Processing {os.path.basename(file_path)}...")
            await rag.process_document_complete(
                file_path=file_path,
                output_dir="./output",
                parse_method="auto"
            )

    # Query - perfect for a student list table
    query = "Extract and display the complete student list as a clean markdown table with all columns and rows."

    print(f"\nQuerying: {query}")
    result = await rag.aquery(query, mode="hybrid")

    print("\n" + "="*50)
    print("FINAL ANSWER:")
    print("="*50)
    print(result)
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main()) 
