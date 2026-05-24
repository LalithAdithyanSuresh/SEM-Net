import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="High-Performance File Upload & Download C2 Server")

# Enable CORS for frontend and cross-origin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Target data directory where downloads are served and uploads are stored
DATA_FOLDER = "/home/cloudai578/data"  # Change this path if needed

# Ensure target folder physically exists
os.makedirs(DATA_FOLDER, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# 1. DOWNLOAD ENDPOINT
# ──────────────────────────────────────────────────────────────────────
@app.get("/download/{filename}")
async def download_file(filename: str):
    # Sanitize the filename to prevent directory traversal
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(DATA_FOLDER, safe_filename)
    
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise HTTPException(
            status_code=404, 
            detail=f"File '{safe_filename}' not found on the server."
        )
    
    return FileResponse(
        path=file_path,
        media_type="application/octet-stream",
        filename=safe_filename
    )

# ──────────────────────────────────────────────────────────────────────
# 2. CHUNKED UPLOAD ENDPOINT
# ──────────────────────────────────────────────────────────────────────
@app.post("/api/upload_chunk")
async def upload_chunk(
    file: UploadFile = File(...),
    filename: str = Form(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...)
):
    # Sanitize filename
    safe_filename = os.path.basename(filename)
    
    # Isolated folder for temporary chunk storage
    temp_dir = os.path.join(DATA_FOLDER, f"temp_{safe_filename}")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Path for this specific chunk
    chunk_path = os.path.join(temp_dir, f"chunk_{chunk_index}")
    
    try:
        with open(chunk_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to write chunk {chunk_index}: {e}"
        )
        
    # Check if all chunks have been received
    received_chunks = len(os.listdir(temp_dir))
    if received_chunks == total_chunks:
        final_file_path = os.path.join(DATA_FOLDER, safe_filename)
        
        # Merge all chunks sequentially
        try:
            with open(final_file_path, "wb") as outfile:
                for i in range(total_chunks):
                    chunk_file = os.path.join(temp_dir, f"chunk_{i}")
                    if not os.path.exists(chunk_file):
                        raise HTTPException(
                            status_code=400, 
                            detail=f"Missing chunk index {i} during file assembly."
                        )
                    with open(chunk_file, "rb") as infile:
                        shutil.copyfileobj(infile, outfile)
            
            # Clean up the temporary chunk folder
            shutil.rmtree(temp_dir)
            print(f"[SERVER] Successfully assembled: {safe_filename}")
            return {"status": "success", "message": f"File {safe_filename} assembled successfully."}
        except Exception as e:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to assemble file: {e}"
            )
            
    return {"status": "success", "message": f"Chunk {chunk_index}/{total_chunks} received."}

# ──────────────────────────────────────────────────────────────────────
# 3. LIST MODEL CHECKPOINTS ENDPOINT
# Returns all .pth files in DATA_FOLDER, sorted by name (iter prefix).
# The session param is accepted for API compatibility but files are flat.
# ──────────────────────────────────────────────────────────────────────
@app.get("/api/models")
async def list_models(session: str = "default"):
    try:
        files = sorted(
            f for f in os.listdir(DATA_FOLDER)
            if f.endswith(".pth") and os.path.isfile(os.path.join(DATA_FOLDER, f))
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not list files: {e}")
    return {"files": files, "session": session}

if __name__ == '__main__':
    import uvicorn
    # Bind to localhost port 8000 (referenced by Nginx proxy pass)
    uvicorn.run(app, host="127.0.0.1", port=8000)
