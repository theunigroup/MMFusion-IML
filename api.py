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
from flask import Flask, request, jsonify
import io
from PIL import Image
import logging
from pythonjsonlogger import jsonlogger
import os
from pathlib import Path
# Initialize the Flask app
app = Flask(__name__)

# Set up JSON logger
def setup_logger(log_level):
    logger = logging.getLogger("flask")
    log_handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    log_handler.setFormatter(formatter)
    logger.addHandler(log_handler)
    logger.setLevel(log_level)
    return logger

# Initialize logger
logger = setup_logger(logging.INFO)

def setup_model(exp_path, ckpt_path, gpu):
    global config  # Declare config as global to modify it
    config = update_config(config, exp_path)

    # Determine the device
    device = f'cuda:{gpu}' if gpu >= 0 else 'cpu'
    np.set_printoptions(formatter={'float': '{: 7.3f}'.format})

    if device.startswith('cuda'):
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = False
        cudnn.deterministic = True
        cudnn.enabled = config.CUDNN.ENABLED

    # Load the model
    modal_extractor = ModalitiesExtractor(config.MODEL.MODALS[1:], config.MODEL.NP_WEIGHTS)
    model = CMNeXtWithConf(config.MODEL)

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

    return device, modal_extractor, model, config


@app.route("/", methods=["POST"])
def calculateDetectionScore():
    try:
        # Extract request parameters
        request_id = str(uuid.uuid4())
        start_time = time.time()

        gpu = int(request.args.get('gpu', -1))
        log = request.args.get('log', 'INFO')
        exp = request.args.get('exp', 'experiments/ec_example_phase2.yaml')
        ckpt = request.args.get('ckpt', 'ckpt/early_fusion_detection.pth')
        
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
        device, modal_extractor, model, config = setup_model(exp, ckpt, gpu)
        
        # Read and preprocess image
        image_stream = request.files['file'].read()
        image = Image.open(io.BytesIO(image_stream)).convert("RGB")
        # Get the current working directory
        current_dir = Path.cwd()
        
        # Construct the full path
        temp_input_path = f"{current_dir}/{request_id}.jpg"
        image.save(temp_input_path)

        # Write to temporary dataset file
        with open('tmp_inf.txt', 'w') as f:
            f.write(temp_input_path + ' None 0\n')

        val = ManipulationDataset('tmp_inf.txt', config.DATASET.IMG_SIZE, train=False)
        val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=config.WORKERS, pin_memory=True)

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
            "file_name": request.files['file'].filename,
            "detection_score": det,
            "execution_time": f"{execution_time:.2f} seconds"
        })

        return jsonify({"detectionScore": det})

    except Exception as e:
        logger.error("Error during inference", extra={
            "request_id": request_id,
            "error": str(e)
        })
        return jsonify({"message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3059)
