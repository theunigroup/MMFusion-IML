import time
import uuid
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from data.datasets import ManipulationDataset
from models.cmnext_conf import CMNeXtWithConf
from models.modal_extract import ModalitiesExtractor
from configs.cmnext_init_cfg import _C as config, update_config
from fastapi import FastAPI, File, UploadFile, Query, Request
from fastapi.responses import JSONResponse
import io
from PIL import Image
import logging
from pythonjsonlogger import jsonlogger
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn

# Initialize the FastAPI app
app = FastAPI()

# Set up JSON logger
def setup_logger(log_level):
    logger = logging.getLogger("uvicorn")
    log_handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    log_handler.setFormatter(formatter)
    logger.addHandler(log_handler)
    logger.setLevel(log_level)
    return logger

# Initialize logger
logger = setup_logger(logging.INFO)

# Middleware to add a unique request ID to each request
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Generate a UUID
        request_id = str(uuid.uuid4())
        # Add the UUID to the request state so it can be accessed later
        request.state.request_id = request_id
        # Log the start time
        request.state.start_time = time.time()
        # Append the request ID to the response headers
        response = await call_next(request)
        response.headers['X-Request-ID'] = request_id
        return response

# Add the middleware to the FastAPI app
app.add_middleware(RequestIDMiddleware)

def setup_model(exp_path, ckpt_path, gpu):
    local_config = update_config(config, exp_path)  # Use local_config instead of config

    # Determine the device
    device = f'cuda:{gpu}' if gpu >= 0 else 'cpu'
    np.set_printoptions(formatter={'float': '{: 7.3f}'.format})

    if device.startswith('cuda'):
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = False
        cudnn.deterministic = True
        cudnn.enabled = local_config.CUDNN.ENABLED

    # Load the model
    modal_extractor = ModalitiesExtractor(local_config.MODEL.MODALS[1:], local_config.MODEL.NP_WEIGHTS)
    model = CMNeXtWithConf(local_config.MODEL)

    # Load model and extractor weights conditionally based on device type
    if gpu >= 0:
        ckpt = torch.load(ckpt_path)
    else:
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))

    model.load_state_dict(ckpt['state_dict'])
    modal_extractor.load_state_dict(ckpt['extractor_state_dict'])

    modal_extractor.to(device)
    model.to(device)
    modal_extractor.eval()
    model.eval()

    return device, modal_extractor, model, local_config

# Initialize models with default parameters
device, modal_extractor, model, local_config = setup_model(
    'experiments/ec_example_phase2.yaml',
    'ckpt/early_fusion_detection.pth',
    0
)

@app.post("/")
async def create_upload_file(
    request: Request,
    file: UploadFile = File(...),
    gpu: int = Query(0, description="GPU device ID, use -1 for CPU"),
    log: str = Query("INFO", description="Logging level in uppercase"),
    exp: str = Query('experiments/ec_example_phase2.yaml', description="Experiment YAML file path"),
    ckpt: str = Query('ckpt/early_fusion_detection.pth', description="Checkpoint file path")
):
    try:
        # Extract request ID and start time
        request_id = request.state.request_id
        start_time = request.state.start_time

        # Update logger level
        logging_level = getattr(logging, log.upper(), logging.INFO)
        logger.setLevel(logging_level)

        # Log request parameters
        logger.info("Received upload request", extra={
            "request_id": request_id,
            "gpu": gpu,
            "log_level": log,
            "exp_path": exp,
            "ckpt_path": ckpt
        })

        # Setup device and models as per query parameters
        device, modal_extractor, model, local_config = setup_model(exp, ckpt, gpu)

        # Read and preprocess image
        image_stream = await file.read()
        image = Image.open(io.BytesIO(image_stream)).convert("RGB")
        temp_input_path = f"{request_id}.jpg"

        # Write to temporary dataset file
        with open('tmp_inf.txt', 'w') as f:
            f.write(temp_input_path + ' None 0\n')

        val = ManipulationDataset('tmp_inf.txt', local_config.DATASET.IMG_SIZE, train=False)
        val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=local_config.WORKERS, pin_memory=True)

        for step, (images, _, masks, lab) in enumerate(val_loader):
            with torch.no_grad():
                images = images.to(device, non_blocking=True)
                masks = masks.squeeze(1).to(device, non_blocking=True)
                modals = modal_extractor(images)

                images_norm = TF.normalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                inp = [images_norm] + modals

                anomaly, confidence, detection = model(inp)
                det = detection.item()

        # Calculate execution time
        execution_time = time.time() - start_time

        # Log results
        logger.info("Inference complete", extra={
            "request_id": request_id,
            "file_name": file.filename,
            "detection_score": det,
            "execution_time": f"{execution_time:.2f} seconds"
        })

        return JSONResponse(content={"detectionScore": det})

    except Exception as e:
        logger.error("Error during inference", extra={
            "request_id": request_id,
            "error": str(e)
        })
        return JSONResponse(status_code=500, content={"message": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3059)
