"""Launch the Model Forge local web server."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8757, reload=False)
