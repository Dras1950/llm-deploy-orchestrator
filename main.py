import uvicorn
from src.api.api import app
from src.config import Config

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=Config.API_PORT)
