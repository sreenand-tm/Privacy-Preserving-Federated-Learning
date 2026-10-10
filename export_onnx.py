import torch
import os
import shutil
from model import UrbanSoundCNN

def export_single_model(checkpoint_path, output_onnx_path):
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found!")
        return False
        
    print(f"Loading {checkpoint_path}...")
    model = UrbanSoundCNN(num_classes=10)
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    dummy_input = torch.randn(1, 1, 128, 173)
    
    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)
    print(f"Exporting to {output_onnx_path}...")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 3: 'time_frames'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"✅ Successfully exported {output_onnx_path}")
    return True

def export_all():
    print("=" * 60)
    print("  ONNX MODEL EXPORTER (5-CLIENT & 8-CLIENT MODELS)")
    print("=" * 60)
    
    # Export 5-client model
    export_single_model("fed/global_model_robust.pth", "web_demo/model_5clients.onnx")
    
    # Export 8-client model
    success_8 = export_single_model("8_train/global_model_robust.pth", "web_demo/model_8clients.onnx")
    
    # Set default model.onnx to 8-client model
    if success_8:
        shutil.copy("web_demo/model_8clients.onnx", "web_demo/model.onnx")
        print("✅ Default web_demo/model.onnx set to 8-client model!")

if __name__ == "__main__":
    export_all()
