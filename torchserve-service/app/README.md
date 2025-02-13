# FasterRCNNHandler - TorchServe Inference Service

## Overview
FasterRCNNHandler is a custom TorchServe inference handler for Faster R-CNN object detection. It supports GPU acceleration, asynchronous processing, and Kafka integration for result streaming.

## Handler Features
- **Device Selection**: Automatically selects GPU, XPU, MPS, or CPU.
- **Asynchronous Processing**: Requests are acknowledged immediately and processed in the background.
- **Kafka Integration**: Results are published to the `torchserve-inference-results-fasterrcnn` topic.
- **Dynamic Image Preprocessing**: Supports multiple input formats (base64, file path, URL).

## Key Methods
### `initialize(self, context)`
- Loads the Faster R-CNN model.
- Configures the Kafka producer.
- Selects the appropriate computing device.

### `handle(self, data, context)`
- Receives requests and immediately acknowledges them.
- Processes requests asynchronously.

### `preprocess(self, data)`
- Extracts image references and converts them to tensors.

### `postprocess(self, data)`
- Packages results for Kafka publication.

## Input Format
Requests should be sent as JSON with an array of image references:
```json
{
  "data": [
    {"path": "image1.jpg", "uuid": "123"},
    {"base64": "iVBORw0KG...", "uuid": "456"},
    {"url": "https://example.com/image.jpg", "uuid": "789"}
  ]
}
```

## Output Format

The service supports two possible output formats:

### Old Format (List of Dictionaries)
```json
[
  {
    "uuid": "abc",
    "detections": [
      {"box": [1345, 184, 2085, 2109], "label": "person", "score": 1.0},
      {"box": [1900, 641, 2444, 1108], "label": "potted plant", "score": 1.0},
      {"box": [1, 260, 562, 1714], "label": "potted plant", "score": 1.0},
      {"box": [1322, 1127, 3112, 1993], "label": "couch", "score": 0.99},
      {"box": [2093, 969, 2260, 1121], "label": "vase", "score": 0.91},
      {"box": [254, 1131, 1254, 1919], "label": "couch", "score": 0.8}
    ]
  }
]
```

### New Format (Dictionary with Image UUID as Keys)
```json
{
  "abc": {
    "detections": [
      {"box": [348, 167, 2464, 3998], "label": "person", "score": 1.0},
      {"box": [334, 1662, 977, 3277], "label": "handbag", "score": 0.83},
      {"box": [631, 2439, 1888, 3661], "label": "tie", "score": 0.81}
    ],
    "size": [4000, 2662]
  },
  "xyz": {
    "detections": [
      {"box": [810, 75, 2363, 3132], "label": "person", "score": 0.99},
      {"box": [31, 2489, 2627, 3970], "label": "bed", "score": 0.99},
      {"box": [16, 1403, 2645, 3757], "label": "bed", "score": 0.82}
    ],
    "size": [4000, 2662]
  }
}
```

## Running the Service
0. Notice the following files: `fasterrcnn_resnet50_fpn_v2_coco-dd69338a.7z.001` and `fasterrcnn_resnet50_fpn_v2_coco-dd69338a.7z.002`, extract them in place using 7zip.
1. (Optionally) Start the Kafka broker.
2. Run the TorchServe container:
   ```sh
   docker-compose up -d
   ```
3. Send an inference request:
   ```sh
   curl -X POST "http://localhost:8080/predictions/fasterrcnn" \
        -H "Content-Type: application/json" \
        -d '{"data": [{"path": "image.jpg", "uuid": "123"}]}'
   ```
