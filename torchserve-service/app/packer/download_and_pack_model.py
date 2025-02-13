# download_and_pack_model.py (For Docker build only)
import subprocess

# import torch
# import torchvision

class ModelPacker:
    def pack_model(self):
        # if not os.path.exists(self.model_path):
        #     print("Model file not found. Downloading...")
        #     try:
        #         weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        #         model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)
        #         torch.save(model.state_dict(), self.model_path)
        #         print("Model downloaded and saved.")
        #     except Exception as e:
        #         print(f"Error downloading model: {e}")
        #         exit(1)  # Exit with error code to stop Docker build

        try:
            print("Creating .mar file...")
            subprocess.run([
                "torch-model-archiver",
                "--model-name", "fasterrcnn",
                "--version", "1.0",
                "--handler", "model/model_handler_runtime.py",
                "--serialized-file", "model/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth",
                "--requirements", "model/requirements.txt",
                "--export-path", "shared/model_store",
                "--archive-format", "no-archive",
                "-f"
            ], check=True)
            print(".mar file created.")
        except subprocess.CalledProcessError as e:
            print(f"Error creating .mar file: {e}")
            exit(1)  # Exit with error code to stop Docker build


if __name__ == "__main__":
    packer = ModelPacker()
    packer.pack_model()
