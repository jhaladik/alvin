# Pac-Man AI Model Deployment

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
