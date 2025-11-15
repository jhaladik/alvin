"""
Local ML Inference Server
Loads trained PyTorch DQN model and serves predictions via REST API
"""
import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dqn_network import DQNNetwork

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global model variable
model = None
device = None

def load_model():
    """Load trained DQN model from checkpoint"""
    global model, device

    print("[+] Loading DQN model...")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Load checkpoint
    checkpoint_path = 'checkpoints/best_dqn_model.pth'
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"DQN checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"  Checkpoint epoch: {checkpoint['epoch']}")
    print(f"  Validation loss: {checkpoint['val_loss']:.2f}")
    print(f"  Validation TD error: {checkpoint['val_td_error']:.2f}")

    # Create DQN model
    model = DQNNetwork(
        input_dim=768,
        hidden_dims=checkpoint['config']['hidden_dims'],
        output_dim=4,
        dropout=0.0  # No dropout for inference
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print("[+] DQN model loaded successfully!")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Model type: DQN (Q-value based, reward-aware)")

    return model

@app.route('/health', methods=['GET'])
@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'device': str(device)
    })

@app.route('/predict', methods=['POST'])
@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict action from feature vector

    Expected JSON:
    {
        "vector": [128-dimensional array],
        "gameState": {...},
        "previousMoves": [...]
    }
    """
    try:
        data = request.json

        # Extract feature vector
        vector = data.get('vector')
        vector_len = len(vector) if vector else 0

        if not vector:
            print(f"[ERROR] No vector provided in request")
            return jsonify({
                'success': False,
                'error': 'No feature vector provided'
            }), 400

        # Handle dimension mismatch - frontend sends 128-dim, model expects 768-dim
        if vector_len == 128:
            # Pad 128-dim vector to 768-dim with zeros
            # Training data used Cloudflare AI embeddings (768-dim)
            # Frontend uses local feature engineering (128-dim)
            vector = vector + [0.0] * (768 - 128)
            print(f"[INFO] Padded 128-dim vector to 768-dim")
        elif vector_len != 768:
            print(f"[ERROR] Wrong vector dimension: got {vector_len}, expected 128 or 768")
            return jsonify({
                'success': False,
                'error': f'Invalid feature vector (expected 128 or 768 dims, got {vector_len})'
            }), 400

        # Convert to tensor
        features = torch.FloatTensor(vector).unsqueeze(0).to(device)

        # Run inference - DQN outputs Q-values
        with torch.no_grad():
            q_values = model(features)

            # Best action = highest Q-value
            predicted_idx = torch.argmax(q_values, dim=1).item()

            # Convert Q-values to probabilities for display (softmax)
            probabilities = torch.softmax(q_values, dim=1)
            confidence = probabilities[0][predicted_idx].item()

        # Map index to action
        actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        action = actions[predicted_idx]

        # Get Q-values and probabilities
        q_vals_dict = {
            actions[i]: float(q_values[0][i].item())
            for i in range(4)
        }
        probs_dict = {
            actions[i]: float(probabilities[0][i].item())
            for i in range(4)
        }

        # Log prediction with Q-values and probabilities
        q_str = ' | '.join([f"{a}: {q_vals_dict[a]:.1f}" for a in actions])
        probs_str = ' | '.join([f"{a}: {probs_dict[a]:.1%}" for a in actions])
        print(f"[DQN] {action} (Q={q_vals_dict[action]:.1f}, conf={confidence:.1%})")
        print(f"  Q-values: {q_str}")
        print(f"  Probs: {probs_str}")

        return jsonify({
            'success': True,
            'prediction': {
                'action': action,
                'confidence': confidence,
                'probabilities': probs_dict,
                'q_values': q_vals_dict,
                'source': 'pytorch_dqn_model'
            },
            'usedML': True,
            'method': 'ml_dqn',
            'model_info': {
                'type': 'dqn',
                'input_dim': 768,
                'output_dim': 4
            }
        })

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        import traceback
        traceback.print_exc()

        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/stats', methods=['GET'])
def stats():
    """Return DQN model statistics"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    return jsonify({
        'model_type': 'DQNNetwork',
        'learning_type': 'Deep Q-Learning (Reward-Based)',
        'input_dim': 768,
        'output_dim': 4,
        'hidden_dims': [256, 256, 128],
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'device': str(device)
    })

def main():
    """Start the DQN inference server"""
    print("=" * 70)
    print("Alvin Pac-Man - DQN Inference Server (Reward-Based Learning)")
    print("=" * 70)
    print()

    # Load model
    try:
        load_model()
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print()
    print("=" * 70)
    print("Server starting on http://localhost:5000")
    print("=" * 70)
    print()
    print("Endpoints:")
    print("  GET  /health   - Health check")
    print("  POST /predict  - Make prediction")
    print("  GET  /stats    - Model statistics")
    print()

    # Start Flask server
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    exit(main())
