
import streamlit as st
import asyncio
import os
import sys
from pathlib import Path

# Add current directory to path to allow imports from raganything package
sys.path.append(str(Path(__file__).parent))

from raganything.raganything import RAGAnything
from raganything.utils import validate_image_file

# Set page config
st.set_page_config(
    page_title="RAG-Anything Chat",
    page_icon="🤖",
    layout="wide"
)


# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Store CONFIG instead of the object
if "rag_config" not in st.session_state:
    st.session_state.rag_config = None

if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False

if "current_file" not in st.session_state:
    st.session_state.current_file = None

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger
from raganything.config import RAGAnythingConfig

from openai import AsyncOpenAI
import numpy as np
import logging

# Configure logging to file
logging.basicConfig(
    filename='debug.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

# @st.cache_resource # Removed caching to prevent event loop mismatch issues with asyncio.run()
def get_rag_instance(working_dir, api_key=None, base_url=None, parser="mineru"):
    """Initialize RAG instance (Non-cached to avoid asyncio loop mismatch)"""
    
    # --- CRITICAL FIX: Clear LightRAG Global Lock Registry ---
    # Streamlit creates a new event loop on every run. LightRAG's global module-level locks 
    # become bound to the old (closed) loop, causing "bound to a different event loop" errors.
    # We must clear them to force recreation in the current loop.
    # --- CRITICAL FIX: Clear LightRAG Global Lock Registry ---
    # Streamlit creates a new event loop on every run. LightRAG's global module-level locks 
    # become bound to the old (closed) loop, causing "bound to a different event loop" errors.
    # We must clear them to force recreation in the current loop.
    try:
        from lightrag.kg import shared_storage
        
        # Reset all lock registries and global state variables
        targets = [
            "_lock_registry", "_async_locks", "_shared_dicts", 
            "_storage_keyed_lock", "_data_init_lock", "_internal_lock",
            "_init_flags", "_update_flags"
        ]
        
        for target in targets:
            if hasattr(shared_storage, target):
                val = getattr(shared_storage, target)
                if isinstance(val, dict):
                    setattr(shared_storage, target, {})
                elif isinstance(val, set):
                    setattr(shared_storage, target, set())
                else:
                    # For single lock objects, we can't easily "reset" them to None 
                    # as the module might expect them to be there. 
                    # But if they are single locks, they are problematic.
                    # Best effort: if it's a lock and we can't reset it, we might be stuck.
                    # However, most of these seem to be dicts or registries based on inspecting.
                    # _storage_keyed_lock seems to be a registry (dict) in newer/some implementations?
                    # Let's assume dicts for registries.
                    pass
        
        # Force re-initialization flag if present
        if hasattr(shared_storage, "_initialized"):
             shared_storage._initialized = False
             
        # Manually clear specific known dicts just in case
        if hasattr(shared_storage, "_lock_registry"):
            shared_storage._lock_registry = {}
        
        logger.info("Aggressively cleared LightRAG global lock registries")
    except ImportError:
        pass
    # ---------------------------------------------------------
    
    # Custom Embedding Implementation to handle 4096 dimensions
    async def custom_openai_embed(texts: list[str], model: str, api_key: str, base_url: str, max_token_size: int = None) -> np.ndarray:
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        try:
            response = await client.embeddings.create(
                input=texts,
                model=model,
                encoding_format="float"
            )
            return np.array([data.embedding for data in response.data])
        except Exception as e:
            logger.error(f"Custom Embedding Failed: {e}")
            raise e

    # Define LLM model function
    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        logger.info(f"LLM Call Prompt (first 200 chars): {prompt[:200]}...")
        try:
            res = openai_complete_if_cache(
                os.environ.get("LLM_MODEL", "gpt-4o-mini"),
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
            logger.info(f"LLM Response: {str(res)[:100]}...") 
            return res
        except Exception as e:
            logger.error(f"LLM Failed: {e}")
            return None

    # Define vision model function
    def vision_model_func(
        prompt,
        system_prompt=None,
        history_messages=[],
        image_data=None,
        messages=None,
        **kwargs,
    ):
        try:
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
            elif image_data:
                return openai_complete_if_cache(
                    os.environ.get("VISION_MODEL", "gpt-4o"),
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=[
                        {"role": "system", "content": system_prompt} if system_prompt else None,
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                            ],
                        } if image_data else {"role": "user", "content": prompt},
                    ],
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )
            else:
                return llm_model_func(prompt, system_prompt, history_messages, **kwargs)
        except Exception as e:
            logger.error(f"Vision Model Failed: {e}")
            return None

    # Define embedding function
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "4096"))
    embedding_model = os.getenv("EMBEDDING_MODEL", "nvidia/nv-embed-v1")

    embedding_func = EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=2048,
        func=lambda texts, max_token_size=None: custom_openai_embed(
            texts,
            model=embedding_model,
            api_key=api_key,
            base_url=base_url,
            max_token_size=max_token_size,
        ),
    )

    # Config object
    config = RAGAnythingConfig(
        working_dir=working_dir,
        parser=parser,
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )
    
    # Match CLI example params for performance and consistency
    lightrag_kwargs = {
        "chunk_token_size": 1024,
        "chunk_overlap_token_size": 100,
        "addon_params": {
            "insert_batch_size": 20, 
        },
         "embedding_func_max_async": 32,
         "llm_model_max_async": 32,
    }
    
    # Pass functions to RAGAnything
    rag = RAGAnything(
        config=config, 
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
        lightrag_kwargs=lightrag_kwargs
    )
    # logger.info(f"RAG Initialized: {rag}")
    return rag

async def process_document_async(rag, file_path, working_dir, skip_ingestion=False):
    """Async wrapper for processing logic"""
    try:
        ext = os.path.splitext(file_path)[1].lower()
        output_dir = os.path.join(working_dir, "output")
        fast_mode_context = None
        
        if skip_ingestion:
            st.info("Found existing storage. Skipping intensive ingestion...")
            
            # --- Smart Context Loading (Same as CLI) ---
            if ext in ['.mp4', '.avi', '.mov', '.mkv']:
                frames_dir = os.path.join(os.path.dirname(file_path), f"{os.path.basename(file_path)}_frames")
                txt_path = os.path.join(frames_dir, "transcript.txt")
                if os.path.exists(txt_path):
                     try:
                        with open(txt_path, "r", encoding="utf-8") as f:
                             content = f.read()
                        if len(content) < 50000:
                             fast_mode_context = content
                             st.success("🚀 Smart Mode: Loaded existing timestamped transcript for context.")
                     except Exception as e:
                         st.warning(f"Could not load existing transcript: {e}")
        else:
            if ext in ['.mp4', '.avi', '.mov', '.mkv']:
                st.info("Processing Video... extracting frames and transcribing.")
                
                frames_dir = os.path.join(os.path.dirname(file_path), f"{os.path.basename(file_path)}_frames")
                txt_path = os.path.join(frames_dir, "transcript.txt")
                
                # --- Frame Extraction ---
                try:
                    import cv2
                    import shutil
                except ImportError:
                    return False, "Error: 'opencv-python' is required for video processing.", None

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
                st.info(f"Extracted {saved_count} frames.")

                # --- Audio Transcription ---
                try:
                    import whisper
                    st.info("Extracting audio for transcription...")
                    model = whisper.load_model("base")
                    result = model.transcribe(file_path)
                    
                    formatted_transcript = []
                    for segment in result["segments"]:
                        start_m, start_s = divmod(int(segment["start"]), 60)
                        end_m, end_s = divmod(int(segment["end"]), 60)
                        time_str = f"[{start_m:02d}:{start_s:02d}-{end_m:02d}:{end_s:02d}]"
                        formatted_transcript.append(f"{time_str} {segment['text']}")
                    
                    transcription = "\n".join(formatted_transcript)
                    
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(f"Audio Transcript of {os.path.basename(file_path)} with Timestamps:\n\n{transcription}")
                    
                    if len(transcription) < 50000:
                        fast_mode_context = transcription
                        st.success("🚀 Smart Mode: Transcript loaded.")
                
                except Exception as e:
                    st.warning(f"Audio transcription failed: {e}")

                st.info("Ingesting processed frames and transcript...")
                
                # Process the FOLDER of frames + transcript
                await rag.process_folder_complete(
                    folder_path=frames_dir,
                    output_dir=output_dir,
                    file_extensions=[".jpg", ".png", ".txt"],
                    recursive=False,
                    max_workers=4
                )
            
            elif ext in ['.xlsx', '.xls', '.csv']:
                st.info("Processing Spreadsheet...")
                await rag.process_document_complete(
                    file_path=file_path,
                    output_dir=output_dir,
                    parse_method="auto"
                )
            else:
                st.info(f"Processing {ext} document...")
                await rag.process_document_complete(
                    file_path=file_path,
                    output_dir=output_dir,
                    parse_method="auto"
                )
                
        return True, "Processing Complete!", fast_mode_context
    except Exception as e:
        return False, str(e), None


# --- UI ---

st.title("🤖 RAG-Anything Chat")

with st.sidebar:
    st.header("Configuration")
    
    # API Settings (Optional override)
    api_key = st.text_input("NVIDIA API Key", type="password", help="Leave empty to use env var")
    
    st.divider()
    
    st.header("Document Input")
    file_path_input = st.text_input("Local File Path", placeholder="/path/to/video.mp4 or document.pdf")
    
    process_btn = st.button("Process Document", type="primary")
    
    final_api_key = api_key if api_key else os.environ.get("LLM_BINDING_API_KEY")
    final_base_url = os.environ.get("LLM_BINDING_HOST")

    if process_btn and file_path_input:
        # Sanitize input path (remove quotes and extra whitespace)
        file_path_input = file_path_input.strip().strip("'").strip('"')
        
        if not os.path.exists(file_path_input):
            st.error(f"File not found: {file_path_input}")
        else:
            # Set up working directory based on filename
            file_name = os.path.basename(file_path_input)
            sanitized_name = "".join([c if c.isalnum() else "_" for c in file_name])
            working_dir = f"./rag_storage/{sanitized_name}"
            
            # Check for existing storage
            skip_ingestion = False
            if os.path.exists(os.path.join(working_dir, "vdb_entities.json")):
                st.success("Existing index found! Using Fast Mode.")
                skip_ingestion = True
            
            # Reset previous context
            st.session_state.fast_mode_context = None

            os.makedirs(working_dir, exist_ok=True)
            
            with st.spinner("Initializing RAG Engine..."):
                try:
                    # Get cached instance
                    rag = get_rag_instance(working_dir, api_key=final_api_key, base_url=final_base_url)
                    
                    # Run async processing
                    success, msg, context = asyncio.run(process_document_async(rag, file_path_input, working_dir, skip_ingestion))
                    
                    if success:
                        # Store CONFIG only
                        st.session_state.rag_config = {
                            "working_dir": working_dir,
                            "api_key": final_api_key,
                            "base_url": final_base_url
                        }
                        if context:
                            st.session_state.fast_mode_context = context
                            
                        st.session_state.processing_complete = True
                        st.session_state.current_file = file_name
                        st.session_state.messages = [] # Reset chat on new file
                        st.success(msg)
                    else:
                        st.error(f"Error: {msg}")
                except Exception as e:
                    st.error(f"Initialization Failed: {e}")

if st.session_state.processing_complete and st.session_state.rag_config:
    st.caption(f"Chatting with: **{st.session_state.current_file}**")
    
    # Re-retrieve the instance using config (will hit cache)
    rag = get_rag_instance(**st.session_state.rag_config)
    # st.write(f"DEBUG: RAG Instance Type: {type(rag)}") # Removed debug
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask about the document..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Fix for potential serialization corruption of internal lightrag object
                    if rag.lightrag is not None and isinstance(rag.lightrag, dict):
                        st.warning("Detected corrupted RAG state (dict reference). Re-initializing...")
                        rag.lightrag = None
                        
                    # Determine System Prompt
                    system_instruction = "You are a helpful assistant."
                    low_prompt = prompt.lower()
                    
                    if any(k in low_prompt for k in ['give', 'extract', 'number', 'details', 'name', 'code', 'date', 'bod', 'address', 'pan', 'id', 'what is']):
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
                    elif any(k in low_prompt for k in ['explain', 'summarize', 'what', 'describe', 'about', 'tell me', 'content', 'when', 'time']):
                         system_instruction = (
                            "You are a helpful assistant. Your Task: Answer based ONLY on the provided context.\n"
                            "If the user gives a length constraint (e.g., '10 words'), prioritize that constraint while keeping the core meaning."
                        )
                    else:
                        system_instruction = (
                            "You are a helpful assistant. You MUST strictly follow the user's formatting instructions."
                            "Do not be verbose if asked to be concise."
                        )
                    
                    # INJECT TRANSCRIPT CONTEXT if available (Smart Mode)
                    raw_context = st.session_state.get("fast_mode_context")
                    if raw_context:
                         system_instruction += f"\n\nFULL TRANSCRIPT CONTEXT:\n{raw_context}\n\n(Use this transcript for accuracy, but also check images/RAG context if needed.)"
                    
                    # Run Query
                    logger.info(f"Executing RAG Query: {prompt}")
                    response = asyncio.run(
                        rag.aquery(prompt, system_prompt=system_instruction)
                    )
                    
                    logger.info(f"RAG Raw Response: {response}")
                    
                    if response:
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        st.error("Received empty response from RAG engine.")
                        logger.error("Empty response received from RAG engine")
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")
else:
    st.info("👈 Please enter a file path and click 'Process Document' to begin.")

