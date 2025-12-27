#!/usr/bin/env python
"""
Example script demonstrating the integration of MinerU parser with RAGAnything

This example shows how to:
1. Process documents with RAGAnything using MinerU parser
2. Perform pure text queries using aquery() method
3. Perform multimodal queries with specific multimodal content using aquery_with_multimodal() method
4. Handle different types of multimodal content (tables, equations) in queries
"""

import os
import argparse
import asyncio
import logging
import logging.config
from pathlib import Path

# Add project root directory to Python path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from raganything import RAGAnything, RAGAnythingConfig
from dotenv import load_dotenv
import static_ffmpeg
static_ffmpeg.add_paths()

load_dotenv(dotenv_path=".env", override=False)


def configure_logging():
    """Configure logging for the application"""
    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "raganything_example.log"))

    print(f"\nRAGAnything example log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")


async def process_with_rag(
    file_path: str,
    output_dir: str,
    api_key: str,
    base_url: str = None,
    working_dir: str = None,
    parser: str = None,
    skip_ingestion: bool = False,
):
    """
    Process document with RAGAnything

    Args:
        file_path: Path to the document
        output_dir: Output directory for RAG results
        api_key: OpenAI API key
        base_url: Optional base URL for API
        working_dir: Working directory for RAG storage
    """
    try:
        # Create RAGAnything configuration
        config = RAGAnythingConfig(
            working_dir=working_dir or "./rag_storage",
            parser=parser,  # Parser selection: mineru or docling
            parse_method="auto",  # Parse method: auto, ocr, or txt
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        # Define LLM model function
        def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return openai_complete_if_cache(
                os.environ.get("LLM_MODEL", "gpt-4o-mini"),
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )

        # Performance usage tips updated:
        # User reported 25m+ processing time. This is due to serial execution of many chunks.
        # Fix: Increase batch size (concurrency) and use a balanced chunk size.
        lightrag_kwargs = {
            "chunk_token_size": 1024, # Larger chunks = fewer total LLM calls
            "chunk_overlap_token_size": 100,
            "addon_params": {
                "insert_batch_size": 20, # Increase concurrency (was 10)
            },
            # Allow more async workers for embedding/LLM
             "embedding_func_max_async": 32,
             "llm_model_max_async": 32,
        }

        # Define vision model function for image processing
        def vision_model_func(
            prompt,
            system_prompt=None,
            history_messages=[],
            image_data=None,
            messages=None,
            **kwargs,
        ):
            # If messages format is provided (for multimodal VLM enhanced query), use it directly
            if messages:
                # NVidia API Restriction: Only 1 image allowed per request
                # Filter messages to ensure only one image is present
                filtered_messages = []
                image_found = False
                
                for msg in messages:
                    if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                        new_content = []
                        for content_part in msg["content"]:
                            if content_part.get("type") == "image_url":
                                if not image_found:
                                    new_content.append(content_part)
                                    image_found = True
                                else:
                                    logger.warning("Limiting to 1 image due to model constraints")
                            else:
                                new_content.append(content_part)
                        
                        filtered_messages.append({"role": "user", "content": new_content})
                    else:
                        filtered_messages.append(msg)
                
                return openai_complete_if_cache(
                    os.environ.get("VISION_MODEL", "gpt-4o"),
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=filtered_messages,
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )
            # Traditional single image format
            elif image_data:
                return openai_complete_if_cache(
                    os.environ.get("VISION_MODEL", "gpt-4o"),
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=[
                        {"role": "system", "content": system_prompt}
                        if system_prompt
                        else None,
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_data}"
                                    },
                                },
                            ],
                        }
                        if image_data
                        else {"role": "user", "content": prompt},
                    ],
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )
            # Pure text format
            else:
                return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

        # Define embedding function - using environment variables for configuration
        embedding_dim = int(os.getenv("EMBEDDING_DIM", "4096"))
        embedding_model = os.getenv("EMBEDDING_MODEL", "nvidia/nv-embed-v1")

        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=2048,
            func=lambda texts, max_token_size=None: openai_embed.func(
                texts,
                model=embedding_model,
                api_key=api_key,
                base_url=base_url,
                max_token_size=max_token_size,
            ),
        )

        # Initialize RAGAnything with new dataclass structure
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
            lightrag_kwargs=lightrag_kwargs,
        )

        # Helper to check extensions
        _, ext = os.path.splitext(file_path)
        
        # State for Smart Fast Mode
        use_fast_mode = False
        fast_mode_context = ""

        # 1. Batch Processing (Folder)
        if os.path.isdir(file_path):
            if skip_ingestion:
                logger.info("Found existing storage. Skipping folder ingestion.")
            else:
                logger.info(f"Detected directory: {file_path}")
                logger.info("Starting batch processing...")
                await rag.process_folder_complete(
                    folder_path=file_path,
                    output_dir=output_dir,
                    file_extensions=[".pdf", ".docx", ".pptx", ".xlsx", ".jpg", ".png", ".txt", ".md"],
                    recursive=True,
                    max_workers=4
                )
            
        # 2. Video Files (Frame Extraction)
        elif ext.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            frames_dir = os.path.join(os.path.dirname(file_path), f"{os.path.basename(file_path)}_frames")
            txt_path = os.path.join(frames_dir, "transcript.txt")

            if skip_ingestion:
                logger.info(f"Detected video file: {file_path}")
                logger.info("Found existing storage. Skipping frame extraction and ingestion...")
                
                # Try to load transcript for context if it exists
                if os.path.exists(txt_path):
                     try:
                        with open(txt_path, "r", encoding="utf-8") as f:
                             content = f.read()
                        if len(content) < 50000:
                             fast_mode_context = content
                             logger.info("🚀 Smart Mode: Loaded existing timestamped transcript for context.")
                     except Exception as e:
                         logger.warning(f"Could not load existing transcript: {e}")
            else:
                logger.info(f"Detected video file: {file_path}")
                logger.info("Extracting key frames for processing...")
                
                try:
                    import cv2
                    import shutil
                except ImportError:
                    logger.error("Error: 'opencv-python' is required for video processing.")
                    return

                # Create frames directory
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
                logger.info(f"Extracted {saved_count} frames to {frames_dir}")

                # --- AUDIO TRANSCRIPTION FOR VIDEO ---
                try:
                    import whisper
                    logger.info("Extracting audio from video for transcription...")
                    # We can pass the video file directly to Whisper!
                    model = whisper.load_model("base")
                    result = model.transcribe(file_path)
                    
                    # Format transcript with Timestamps to enable Time <-> Event queries
                    formatted_transcript = []
                    for segment in result["segments"]:
                        start_m, start_s = divmod(int(segment["start"]), 60)
                        end_m, end_s = divmod(int(segment["end"]), 60)
                        time_str = f"[{start_m:02d}:{start_s:02d}-{end_m:02d}:{end_s:02d}]"
                        formatted_transcript.append(f"{time_str} {segment['text']}")
                    
                    transcription = "\n".join(formatted_transcript)
                    
                    # Save transcript to the FRAMES directory so it gets indexed with images
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(f"Audio Transcript of {os.path.basename(file_path)} with Timestamps:\n\n{transcription}")
                    
                    logger.info(f"Saved timestamped transcript to {txt_path}")
                    
                    # Load for System Prompt Injection (Smart Hybrid Mode)
                    if len(transcription) < 50000:
                        fast_mode_context = transcription
                        logger.info("🚀 Smart Mode: Timestamped transcript loaded for time-aware context.")
                
                except Exception as e:
                    logger.warning(f"Audio transcription failed (Video might have no audio): {e}")

                logger.info("Processing frames and audio text as a document folder...")
                
                # Process the folder of frames + transcript
                # Use process_folder_complete as frames_dir is a directory
                await rag.process_folder_complete(
                    folder_path=frames_dir,
                    output_dir=output_dir,
                    file_extensions=[".jpg", ".png", ".txt"],
                    recursive=False,
                    max_workers=4
                )

        # 3. Audio Files (Whisper Transcription)
        elif ext.lower() in ['.mp3', '.wav', '.m4a']:
            txt_path = os.path.join(os.path.dirname(file_path), f"{os.path.basename(file_path)}_transcription.txt")

            if skip_ingestion:
                logger.info(f"Detected audio file: {file_path}")
                logger.info("Found existing storage. Skipping transcription and ingestion...")
                if os.path.exists(txt_path):
                     try:
                        with open(txt_path, "r", encoding="utf-8") as f:
                             transcription = f.read()
                        if len(transcription) < 50000:
                            logger.info("🚀 Smart Mode: Transcript loaded for instant, accurate chat.")
                            use_fast_mode = True
                            fast_mode_context = transcription
                     except Exception as e:
                         logger.warning(f"Could not load existing transcript: {e}")
            else:
                logger.info(f"Detected audio file: {file_path}")
                logger.info("Loading Whisper model for transcription...")
                
                try:
                    import whisper
                except ImportError:
                    logger.error("Error: 'openai-whisper' is required for audio processing.")
                    return

                # Transcribe
                model = whisper.load_model("base")
                logger.info("Transcribing audio (this may take a while)...")
                result = model.transcribe(file_path)
                
                # Format transcript with Timestamps
                formatted_transcript = []
                for segment in result["segments"]:
                    start_m, start_s = divmod(int(segment["start"]), 60)
                    end_m, end_s = divmod(int(segment["end"]), 60)
                    time_str = f"[{start_m:02d}:{start_s:02d}-{end_m:02d}:{end_s:02d}]"
                    formatted_transcript.append(f"{time_str} {segment['text']}")
                
                transcription = "\n".join(formatted_transcript)
                
                logger.info("Transcription complete!")
                
                # Save to text file
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(f"Transcription of {os.path.basename(file_path)} with Timestamps:\n\n")
                    f.write(transcription)
                
                logger.info(f"Saved transcription to {txt_path}")
                logger.info(f"Saved transcription to {txt_path}")
                logger.info("Processing transcribed text...")
                
                # check size for Fast Mode
                if len(transcription) < 50000: # Approx 50KB / 12k tokens
                    logger.info("🚀 Smart Mode: Transcript is short. Skipping RAG indexing for instant, accurate chat.")
                    use_fast_mode = True
                    fast_mode_context = transcription
                else:
                    await rag.process_document_complete(
                        file_path=txt_path,
                        output_dir=output_dir,
                        parse_method="auto"
                    )

        # 3.5 Excel Files (Pandas Native Processing) - Bypasses LibreOffice
        elif ext.lower() in ['.xlsx', '.xls']:
            txt_path = os.path.join(os.path.dirname(file_path), f"{os.path.basename(file_path)}_converted.txt")
            
            if skip_ingestion:
                logger.info(f"Detected Excel file: {file_path}")
                logger.info("Found existing storage. Skipping conversion and ingestion...")
                if os.path.exists(txt_path):
                     try:
                        with open(txt_path, "r", encoding="utf-8") as f:
                             full_text = f.read()
                        if len(full_text) < 50000:
                            logger.info("🚀 Smart Mode: Excel content loaded for instant, accurate chat.")
                            use_fast_mode = True
                            fast_mode_context = full_text
                     except Exception as e:
                         logger.warning(f"Could not load existing converted excel: {e}")
            else:
                logger.info(f"Detected Excel file: {file_path}")
                logger.info("Processing Excel file with Pandas (Native)...")
                
                try:
                    import pandas as pd
                    # Read all sheets
                    xls = pd.ExcelFile(file_path)
                    markdown_output = []
                    markdown_output.append(f"# Excel Document: {os.path.basename(file_path)}\n")
                    
                    for sheet_name in xls.sheet_names:
                        markdown_output.append(f"\n## Sheet: {sheet_name}\n")
                        df = pd.read_excel(file_path, sheet_name=sheet_name)
                        # Convert to markdown table
                        markdown_output.append(df.to_markdown(index=False))
                        markdown_output.append("\n")
                    
                    full_text = "\n".join(markdown_output)
                    
                    # Save as converted text file
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(full_text)
                        
                    logger.info(f"Converted Excel to Markdown text at: {txt_path}")
                    
                     # Check size for Fast Mode
                    if len(full_text) < 50000:
                        logger.info("🚀 Smart Mode: Excel content is small. Skipping RAG indexing for instant, accurate chat.")
                        use_fast_mode = True
                        fast_mode_context = full_text
                    else:
                        await rag.process_document_complete(
                            file_path=txt_path,
                            output_dir=output_dir,
                            parse_method="auto"
                        )

                except Exception as e:
                    logger.error(f"Pandas processing failed: {e}")
                    # Fallback to standard (which will likely fail if no LibreOffice, but good to try)
                    await rag.process_document_complete(
                        file_path=file_path, output_dir=output_dir, parse_method="auto"
                    )

        elif ext.lower() in ['.txt', '.md']:
             # Check explicitly for text files to enable Fast Mode
             try:
                 with open(file_path, 'r', encoding='utf-8') as f:
                     content = f.read()
                 if len(content) < 50000:
                     logger.info("🚀 Smart Mode: File is short text. Skipping RAG indexing for instant, accurate chat.")
                     use_fast_mode = True
                     fast_mode_context = content
                     
                 if not use_fast_mode: # Only ingest if not fast mode (or if fast mode but forcing ingest - but here fast mode skips ingest implicitly by context loading)
                      if skip_ingestion:
                            logger.info("Found existing storage. Skipping ingestion...")
                      else:
                         await rag.process_document_complete(
                            file_path=file_path, output_dir=output_dir, parse_method="auto"
                         )
             except Exception as e:
                 logger.warning(f"Could not read text for Smart Mode check: {e}")
                 if skip_ingestion:
                        logger.info("Found existing storage. Skipping ingestion...")
                 else:
                     await rag.process_document_complete(
                        file_path=file_path, output_dir=output_dir, parse_method="auto"
                     )
        
        # 4. Standard File Processing (PDFs, etc - hard to extract text without parser so we rely on RAG)
        else:
            if skip_ingestion:
                 logger.info(f"Found existing storage for {os.path.basename(file_path)}. Skipping ingestion...")
            else:
                logger.info(f"Processing {os.path.basename(file_path)}...")
                await rag.process_document_complete(
                    file_path=file_path, output_dir=output_dir, parse_method="auto"
                )

        # Example queries - demonstrating different query approaches
        logger.info("\nQuerying processed document:")

        # 1. Pure text queries using aquery()
        text_queries = [
            "What is the main content of the document?",
            "What are the key topics discussed?",
        ]

        # Interactive Query Mode
        logger.info("\n" + "="*50)
        logger.info("Interactive Mode: Enter your queries below (type 'exit' to quit)")
        logger.info("="*50 + "\n")

        while True:
            try:
                user_input = input("\nEnter your query: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    logger.info("Exiting interactive mode.")
                    break
                
                if not user_input:
                    continue

                logger.info(f"Processing query: {user_input}")
                
                # Check for multimodal triggers (simple keyword based for demo)
                # In a real app, you might have a more sophisticated way to attach content
                if "compare" in user_input.lower() and "table" in user_input.lower():
                     logger.info("Detected possible table comparison request (Demo)")
                     # Keeping the demo table capability available if requested specifically
                     # This effectively hides it unless asked
                
                # Standard query (text + auto-retrieved context)
                # This handles both pure text and VLM enhanced queries automatically
                
                # Check if it's a summary/explanation request vs a strict constraint request
                # Expanded keywords to catch vague queries like "about video"
                if any(k in user_input.lower() for k in ['explain', 'summarize', 'what', 'describe', 'about', 'tell me', 'content', 'when', 'time']):
                     system_instruction = (
                        "You are a helpful assistant. Your Task: Answer based ONLY on the provided context (transcripts, extracted entities). "
                        "Do not use outside knowledge. "
                        "The transcript contains TIMESTAMPS in [MM:SS] format.\n"
                        "- If asked 'When did X happen?', provide the specific timestamp range.\n"
                        "- If asked 'What happened at 00:10?', describe the event at that time.\n"
                        "- If the user asks 'about video' or 'explain', summarize the main topic.\n"
                        "Do not define what a video is. Tell me what THIS video is about."
                        "If the user gives a length constraint (e.g., '10 words'), prioritize that constraint while keeping the core meaning."
                    )
                
                # Check for Extraction/Specific Detail requests
                if any(k in user_input.lower() for k in ['give', 'extract', 'number', 'details', 'name', 'code', 'date', 'bod', 'address', 'pan', 'id']):
                    system_instruction = (
                        "You are a Document Analysis Assistant.\n"
                        "TASK: The user has provided an image/document they OWN and needs to EXTRACT specific information from it.\n"
                        "INSTRUCTIONS:\n"
                        "1. Extract the requested details EXACTLY as they appear in the visual/text context.\n"
                        "2. Do NOT refuse to provide personal information found in the document, as the user is the owner.\n"
                        "3. If the image is blurry, do your best to decipher the characters and mark uncertain parts with (?).\n"
                        "4. Output ONLY the requested information if the user asks for a specific field (e.g., just the number).\n"
                        "5. If 'bod' is asked, it likely means 'Date of Birth' (DOB)."
                    )
                else:
                    system_instruction = (
                        "You are a helpful assistant. You MUST strictly follow the user's formatting instructions "
                        "(e.g., word count, list format, tone). If the user asks for 10 words, give exactly 10 words. "
                        "Do not be verbose if asked to be concise."
                    )

                if use_fast_mode:
                    # Direct Context Injection (Fast Text Mode)
                    fast_system = system_instruction + "\n\nDOCUMENT CONTEXT:\n" + fast_mode_context
                    result = await llm_model_func(user_input, system_prompt=fast_system)
                else:
                    # Standard RAG Query (Hybrid Mode)
                    # If we have a transcript loaded (even if in slow mode), inject it!
                    if fast_mode_context:
                         system_instruction += f"\n\nFULL TRANSCRIPT CONTEXT:\n{fast_mode_context}\n\n(Use this transcript for accuracy, but also check images/RAG context if needed.)"

                    result = await rag.aquery(user_input, mode="hybrid", system_prompt=system_instruction)
                
                print("\n" + "-"*30)
                print(f"Answer:\n{result}")
                print("-"*30 + "\n")

            except KeyboardInterrupt:
                logger.info("\nInterrupted by user. Exiting.")
                break
            except Exception as e:
                logger.error(f"Error during query: {str(e)}")

    except Exception as e:
        logger.error(f"Error processing with RAG: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())


def main():
    """Main function to run the example"""
    parser = argparse.ArgumentParser(description="MinerU RAG Example")
    parser.add_argument("file_path", help="Path to the document to process")
    parser.add_argument(
        "--working_dir", "-w", default="./rag_storage", help="Working directory path"
    )
    parser.add_argument(
        "--output", "-o", default="./output", help="Output directory path"
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLM_BINDING_API_KEY"),
        help="OpenAI API key (defaults to LLM_BINDING_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BINDING_HOST"),
        help="Optional base URL for API",
    )
    parser.add_argument(
        "--parser",
        default=os.getenv("PARSER", "mineru"),
        help="Optional base URL for API",
    )

    args = parser.parse_args()

    # Check if API key is provided
    if not args.api_key:
        logger.error("Error: OpenAI API key is required")
        logger.error("Set api key environment variable or use --api-key option")
        return

    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # Auto-isolate storage per file if using default directory
    working_dir = args.working_dir
    file_name = os.path.basename(args.file_path)
    if working_dir == "./rag_storage":
        safe_name = "".join([c if c.isalnum() else "_" for c in file_name])
        working_dir = os.path.join(working_dir, safe_name)
        logger.info(f"Auto-isolating storage to: {working_dir}")

    # Check if we can skip processing
    skip_ingestion = False
    if os.path.exists(os.path.join(working_dir, "vdb_entities.json")):
         logger.info(f"Found existing storage in {working_dir}. Skipping ingestion logic checks.")
         skip_ingestion = True

    # Process with RAG
    asyncio.run(
        process_with_rag(
            args.file_path,
            args.output,
            args.api_key,
            args.base_url,
            working_dir,
            args.parser,
            skip_ingestion=skip_ingestion,
        )
    )



if __name__ == "__main__":
    # Configure logging first
    configure_logging()

    print("RAGAnything Example")
    print("=" * 30)
    print("Processing document with multimodal RAG pipeline")
    print("=" * 30)

    main()
