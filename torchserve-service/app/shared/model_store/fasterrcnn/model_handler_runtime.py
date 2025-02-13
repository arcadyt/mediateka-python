import base64
import io
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import requests
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from confluent_kafka import Producer
from ts.torch_handler.base_handler import BaseHandler


class FasterRCNNHandler(BaseHandler):
    def __init__(self):
        super().__init__()

        # this would be set in initialize, once context will become available.
        self.producer = None
        self.batchSizePerImageResolution = None

        # Mapping of class indices to human-readable labels
        self.mapping = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign',
                        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
                        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack',
                        'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee',
                        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
                        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                        'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
                        'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
                        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        self.logger = logging.getLogger(__name__)
        self.transform = lambda img: T.ToTensor()(img).to(self.device)
        self.executor = ThreadPoolExecutor(max_workers=1)  # Adjust worker count based on your needs
        self.RESULTS_TOPIC = 'torchserve-inference-results-fasterrcnn'

    def initialize(self, context):
        self.logger.info("! FasterRCNN Handler: Initialize started !")

        self.logger.info("! FasterRCNN Handler: Initialize started !")

        try:
            self.producer = Producer({
                'bootstrap.servers': 'kafka:9092',
                'client.id': 'torchserve-service-fasterrcnn'
            })
        except Exception as e:
            self.logger.error(f"! Failed to connect to kafka, reason: {e} !")

        # pick device
        properties = context.system_properties
        if torch.cuda.is_available() and properties.get("gpu_id") is not None:
            self.map_location = "cuda"
            self.device = torch.device(
                self.map_location + ":" + str(properties.get("gpu_id"))
            )
        elif (
                os.environ.get("TS_IPEX_GPU_ENABLE", "false") == "true"
                and properties.get("gpu_id") is not None
                and torch.xpu.is_available()
        ):
            self.map_location = "xpu"
            self.device = torch.device(
                self.map_location + ":" + str(properties.get("gpu_id"))
            )
            torch.xpu.device(self.device)
        elif torch.backends.mps.is_available() and properties.get("gpu_id") is not None:
            self.map_location = "mps"
            self.device = torch.device("mps")
        elif self._isXLAAvailable():
            self.logger.warning(f"! FasterRCNN Handler: @@@XLA AVAILABLE, ADD LOGIC@@@ !")
        else:
            self.map_location = "cpu"
            self.device = torch.device(self.map_location)

        self.logger.debug(f"! FasterRCNN Handler: properties: [{properties}] !")
        self.batchSizePerImageResolution = properties.get('batch_size')

        # Load model weights
        try:
            self.logger.info("! FasterRCNN Handler: Loading model started !")
            self.model = models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=False, progress=False,
                                                                     box_score_thresh=0.8, box_detections_per_img=10)
            self.logger.debug("! FasterRCNN Handler: Loading model loaded without weights !")
            model_path = self._getModelPtPath(context)
            self.logger.debug(f"! FasterRCNN Handler: Loading weights path found {model_path} !")
            state_dict = torch.load(model_path, weights_only=True)
            self.logger.debug("! FasterRCNN Handler: Loaded weights !")
            self.model.to(self.device)
            self.logger.debug("! FasterRCNN Handler: moving model to device !")
            self.model.load_state_dict(state_dict)
            self.logger.debug("! FasterRCNN Handler: Pairing model with weights !")
            self.model.eval()
            self.logger.info("! Model initialized successfully !")
        except Exception as e:
            self.logger.error(f"! Failed to load model weights: {e} !")
            raise RuntimeError("No model weights could be loaded")

    def _getModelPtPath(self, context):
        model_dir = context.system_properties.get("model_dir")
        serialized_file = context.manifest["model"]["serializedFile"]
        self.logger.debug(f"! FasterRCNN Handler: Initialize: serialized_file:[{serialized_file}] !")
        model_path = os.path.join(model_dir, serialized_file)
        self.logger.debug(f"! FasterRCNN Handler: Initialize: model_path:[{model_path}] !")
        return model_path

    def _isXLAAvailable(self):
        try:
            import torch_xla.core.xla_model as xm
            XLA_AVAILABLE = True
        except ImportError as error:
            XLA_AVAILABLE = False
        return XLA_AVAILABLE

    def handle(self, data, context):
        """
        Main handle method for asynchronous processing. Acknowledges the request immediately and processes it in the background.
        """
        self.logger.info(f"! FasterRCNN Handler: Handle method started calculation_uuid:{context.request_ids} !")
        calculation_uuid = self._get_calculation_id(context)

        # Immediate acknowledgment with HTTP 202-like response
        response = [{"status": "Request accepted for asynchronous processing", "calculation_uuid": calculation_uuid}]

        # Process the request asynchronously
        self.executor.submit(self._process_request, data, context)

        # Return acknowledgment to the client
        return response

    def _get_calculation_id(self, context):
        return context.request_ids[0]

    def _process_request(self, data, context):
        """
        Processes the request asynchronously by calling the parent's handle method and sending the result to Kafka.
        """
        calculation_id = self._get_calculation_id(context)

        try:
            self.logger.info(
                f"! FasterRCNN Handler: Async request of calculation_uuid: {calculation_id} started !")
            # Use the parent's handle method for full processing (preprocess -> inference -> postprocess)
            output = super().handle(data, context)

            self._publish_to_kafka(calculation_id, output)

            self.logger.info(
                f"! Async processing of calculation_uuid: {calculation_id} just completed successfully {json.dumps(output)} !")
        except Exception as e:
            self.logger.error(
                f"! Async processing of calculation_uuid: {calculation_id} failed. Error: {str(e)} !")

    def _publish_to_kafka(self, calculation_id, output):
        # Prepare the message for Kafka
        kafka_message = {
            'calculation_id': calculation_id,
            'output': output
        }
        if self.producer is not None:
            self.producer.produce(
                topic=self.RESULTS_TOPIC,
                key=calculation_id,
                value=json.dumps(kafka_message).encode('utf-8'),
            )
        else:
            self.logger.warning("! Skipping sending out result to kafka (failed to initialize, restart to retry) !")

    def preprocess(self, data):
        self.logger.info(f"! FasterRCNN Handler: Preprocess called with {str(data)[:200]} !")

        try:
            image_refs = self._extract_input_image_refs(data)

            self.logger.debug(f"! Received {len(image_refs)} images for processing !")

            # Process each image entry
            processed_images = {}
            image_uuids = {}

            for image_ref in image_refs:
                self._validate_image_ref(image_ref)

                # Load the image and UUID
                processed_img, uuid = self._load_image(image_ref)

                # Get image size (e.g., height, width)
                img_size = tuple(processed_img.shape[-2:])  # (height, width)

                # Group images by size
                if img_size not in processed_images:
                    processed_images[img_size] = []
                    image_uuids[img_size] = []
                processed_images[img_size].append(processed_img)
                image_uuids[img_size].append(uuid)

            # Convert each size group into a tensor
            grouped_tensors = {
                size: torch.cat(img_list) for size, img_list in processed_images.items()
            }

            self.logger.info("! Preprocessing completed successfully !")
            return grouped_tensors, image_uuids

        except Exception as e:
            self.logger.error(f"! Preprocessing error: {str(e)} !")
            raise

    def _validate_image_ref(self, image_ref):
        # Validate required keys
        if 'uuid' not in image_ref:
            raise ValueError("! Each image object must contain a 'uuid' key !")
        if not any(key in image_ref for key in ['base64', 'path', 'url']):
            raise ValueError(
                f"! Each image object must contain one of 'base64', 'path', or 'url': {image_ref} !")

    def _extract_input_image_refs(self, data):
        """
        Expected input in postman must like -
        {"data":[{"path":"abc","uuid":"xyz"},{"base64":"012","uuid":"345"},{"url":"www","uuid":"987"]}}
        """
        # TorchServe sends the input as a list; extract the first item
        if not isinstance(data, list) or not data:
            raise ValueError("! Input data must be a non-empty list !")
        # Extract the 'body' from the first item in the list
        input_body = data[0]
        if not isinstance(input_body, dict) or 'body' not in input_body:
            raise ValueError("! Missing 'body' key in input JSON !")
        # Extract the 'data' key from within the 'body'
        body = input_body['body']
        if not isinstance(body, dict) or 'data' not in body:
            raise ValueError("! 'body' must contain a 'data' key !")
        image_refs = body['data']  # Extract the 'data' array -- torchserve is really strict on the "data" name here!
        if not isinstance(image_refs, list) or not image_refs:
            raise ValueError("! The 'data' key must contain a non-empty list of image objects !")
        return image_refs

    def _load_image(self, image_data):
        """
        Load image from base64, URL, or file path
        """
        try:
            # If base64 encoded
            if 'base64' in image_data:
                image_bytes = base64.b64decode(image_data['base64'])
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            # If URL
            elif 'url' in image_data:
                response = requests.get(image_data['url'])
                image = Image.open(io.BytesIO(response.content)).convert('RGB')

            # If file path
            elif 'path' in image_data:
                image = Image.open(image_data['path']).convert('RGB')

            else:
                raise ValueError("! Invalid image source !")

            return self.transform(image).unsqueeze(0), image_data.get('uuid')
        except Exception as e:
            self.logger.error(f"! Image loading error: {str(e)} !")
            raise

    def inference(self, processed_data, *args, **kwargs):
        """
        Run inference with detailed logging and batch processing
        """
        self.logger.info("! FasterRCNN Handler: Inference started !")

        try:
            # Unpack grouped tensors and UUIDs
            grouped_tensors, image_uuids = processed_data
            all_predictions = {}
            all_uuids = {}

            # Run inference for each size group
            for img_size, images in grouped_tensors.items():
                self.logger.info(f"! Inferencing {images.size(0)} images of size {img_size} !")

                # Initialize list to store predictions for current size group
                size_predictions = []

                # Process in batches
                for i in range(0, images.size(0), self.batchSizePerImageResolution):
                    batch_images = images[i:i + self.batchSizePerImageResolution]
                    self.logger.info(f"! Running inference on batch of {batch_images.size(0)} images !")

                    with torch.no_grad():
                        batch_predictions = self.model(batch_images)

                    size_predictions.extend(batch_predictions)

                # Store all predictions and UUIDs for current size group
                all_predictions[img_size] = size_predictions
                all_uuids[img_size] = image_uuids[img_size]

            self.logger.info("! Inference completed successfully !")
            return all_predictions, all_uuids

        except Exception as e:
            self.logger.error(f"! Inference error: {str(e)} !")
            raise

    def postprocess(self, data):
        """
        Postprocess predictions with comprehensive logging and batch consistency.
        """
        self.logger.info("! FasterRCNN Handler: Postprocess started !")

        try:
            # Unpack grouped predictions and UUIDs
            grouped_predictions, grouped_uuids = data

            batch_output = self._collect_batch_results_new(grouped_predictions, grouped_uuids)
            # batch_output = self._collect_batch_results_old(grouped_predictions, grouped_uuids)

            # Log batch output size
            self.logger.info(f"! Postprocess generated {len(batch_output)} outputs !")

            # Return the batch output with consistent size
            return batch_output

        except Exception as e:
            self.logger.error(f"! Postprocess error: {str(e)} !")
            raise

    def _collect_batch_results_new(self, grouped_predictions, grouped_uuids):
        """
        {"uuid-123":{"detections":[{"box":[100,200,300,400],"label":"person","score":0.95}]},"uuid-456":{"detections":[]}}
        """
        batch_output = {}  # Changed to dictionary instead of list
        # Process each size group
        for img_size, predictions in grouped_predictions.items():
            uuids = grouped_uuids[img_size]

            # Ensure predictions and UUIDs have the same batch size
            if len(predictions) != len(uuids):
                raise ValueError(
                    f"! Batch size mismatch for size {img_size}: {len(predictions)} predictions, {len(uuids)} UUIDs !")

            for idx, image_predictions in enumerate(predictions):
                # Filter predictions by confidence threshold
                outputs = [
                    {
                        'box': [round(coord) for coord in box.tolist()],
                        'label': self.mapping[label.item()],
                        'score': round(float(score.item()), 2)
                    }
                    for box, label, score in zip(
                        image_predictions['boxes'],
                        image_predictions['labels'],
                        image_predictions['scores']
                    )
                ]

                # Add entry to batch_output dictionary using UUID as key
                batch_output[uuids[idx]] = {
                    'detections': outputs,
                    'size': img_size
                }
        return batch_output

    def _collect_batch_results_old(self, grouped_predictions, grouped_uuids):
        batch_output = []
        # Process each size group
        for img_size, predictions in grouped_predictions.items():
            uuids = grouped_uuids[img_size]

            # Ensure predictions and UUIDs have the same batch size
            if len(predictions) != len(uuids):
                raise ValueError(
                    f"! Batch size mismatch for size {img_size}: {len(predictions)} predictions, {len(uuids)} UUIDs !")

            for idx, image_predictions in enumerate(predictions):
                # Filter predictions by confidence threshold
                outputs = [
                    {
                        'box': [round(coord) for coord in box.tolist()],
                        'label': self.mapping[label.item()],
                        'score': round(float(score.item()), 2)
                    }
                    for box, label, score in zip(
                        image_predictions['boxes'],
                        image_predictions['labels'],
                        image_predictions['scores']
                    )
                ]

                # Include UUID for this image
                batch_output.append({
                    'uuid': uuids[idx],
                    'detections': outputs,
                    'size': img_size  # Optionally include the image size for debugging or grouping
                })
        return batch_output
