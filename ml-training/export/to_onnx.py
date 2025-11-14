"""
Export trained PyTorch model to ONNX format
For deployment to Cloudflare Workers or browser
"""
import torch
import onnx
import onnxruntime as ort
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.behavioral_cloning import PacManPolicyNetwork


def export_to_onnx(checkpoint_path, output_path, opset_version=12):
    """
    Export PyTorch model to ONNX

    Args:
        checkpoint_path: Path to .pth checkpoint
        output_path: Path to save .onnx file
        opset_version: ONNX opset version
    """
    print("=" * 70)
    print("Exporting Model to ONNX")
    print("=" * 70)
    print()

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Get model config
    config = checkpoint.get('config', {})
    hidden_dims = config.get('hidden_dims', [256, 256, 128])
    dropout = config.get('dropout', 0.3)

    # Create model
    model = PacManPolicyNetwork(
        input_dim=128,
        hidden_dims=hidden_dims,
        output_dim=4,
        dropout=dropout
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"[+] Model loaded")
    print(f"  Validation accuracy: {checkpoint.get('val_acc', 0):.2f}%")
    print(f"  Architecture: {hidden_dims}")
    print()

    # Create dummy input
    dummy_input = torch.randn(1, 128)

    # Export
    print(f"Exporting to ONNX (opset {opset_version})...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"[+] Exported to: {output_path}")

    # Verify ONNX model
    print("\nVerifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("[+] ONNX model is valid")

    # Test with ONNX Runtime
    print("\nTesting with ONNX Runtime...")
    ort_session = ort.InferenceSession(output_path)

    # Test inference
    test_input = np.random.randn(1, 128).astype(np.float32)
    ort_inputs = {ort_session.get_inputs()[0].name: test_input}
    ort_outputs = ort_session.run(None, ort_inputs)

    print(f"[+] Inference successful")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {ort_outputs[0].shape}")
    print(f"  Output: {ort_outputs[0]}")

    # Compare PyTorch vs ONNX
    with torch.no_grad():
        torch_output = model(torch.from_numpy(test_input))
        torch_output = torch_output.numpy()

    diff = np.abs(torch_output - ort_outputs[0]).max()
    print(f"\nMax difference PyTorch vs ONNX: {diff:.8f}")

    if diff < 1e-5:
        print("[+] Outputs match!")
    else:
        print("[!] Warning: Outputs differ slightly")

    # Print model info
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nModel info:")
    print(f"  File size: {file_size:.2f} MB")
    print(f"  Input: (batch_size, 128) - float32")
    print(f"  Output: (batch_size, 4) - float32")

    return output_path


def create_tensorflowjs_model(checkpoint_path, output_dir):
    """
    Alternative: Export to TensorFlow.js format
    (Requires tensorflowjs pip package)
    """
    print("\nNote: TensorFlow.js export requires additional setup")
    print("  pip install tensorflowjs")
    print("  Use pytorch_to_keras converter + tensorflowjs_converter")


def create_deployment_package(onnx_path):
    """Create deployment package with model and instructions"""
    print("\n" + "=" * 70)
    print("Creating Deployment Package")
    print("=" * 70)

    # Create README for deployment
    readme = """# Pac-Man AI Model Deployment

## Model Information
- Format: ONNX
- Input: (batch_size, 128) float32 - game state features
- Output: (batch_size, 4) float32 - action logits

## Actions
0: UP
1: DOWN
2: LEFT
3: RIGHT

## Deployment Options

### Option 1: ONNX Runtime in Browser
```javascript
// Use ONNX Runtime Web
import * as ort from 'onnxruntime-web';

const session = await ort.InferenceSession.create('model.onnx');
const input = new ort.Tensor('float32', features, [1, 128]);
const outputs = await session.run({input: input});
const action = argmax(outputs.output.data);
```

### Option 2: Cloudflare Workers (if supported)
Check Cloudflare Workers AI documentation for ONNX support.

### Option 3: Convert to TensorFlow.js
For better browser support, convert to TFjs format.

## Usage
1. Extract features from game state (128-dim vector)
2. Run inference to get action logits
3. Apply softmax and argmax to get action
4. Execute action in game
"""

    readme_path = os.path.join(os.path.dirname(onnx_path), 'DEPLOYMENT.md')
    with open(readme_path, 'w') as f:
        f.write(readme)

    print(f"[+] Created {readme_path}")
    print("\n[+] Deployment package ready!")


def main():
    checkpoint_path = "checkpoints/best_model.pth"
    output_path = "export/pacman_model.onnx"

    if not os.path.exists(checkpoint_path):
        print(f"[-] Checkpoint not found: {checkpoint_path}")
        print("\nRun training first:")
        print("  python training/train.py")
        return 1

    # Create export directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Export to ONNX
    onnx_path = export_to_onnx(checkpoint_path, output_path)

    # Create deployment package
    create_deployment_package(onnx_path)

    print("\n" + "=" * 70)
    print("[+] Export Complete!")
    print("=" * 70)
    print(f"\nModel saved to: {output_path}")
    print("\nNext steps:")
    print("  1. Test model with sample data")
    print("  2. Integrate into Cloudflare Worker or frontend")
    print("  3. Deploy and verify performance")

    return 0


if __name__ == "__main__":
    exit(main())
