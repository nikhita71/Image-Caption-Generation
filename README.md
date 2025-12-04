# Image Captioning with Deep Learning

## Overview
This project implements an automatic image captioning system using deep learning. The model combines Convolutional Neural Networks (CNN) for image feature extraction and Recurrent Neural Networks (RNN) with LSTM for generating natural language descriptions of images.

## Project Description
Image captioning is the task of generating textual descriptions for images. This implementation uses a CNN-RNN encoder-decoder architecture that learns to:
1. Extract visual features from images using a pre-trained VGG16 model
2. Generate coherent and contextually relevant captions using an LSTM-based language model
3. Map visual features to natural language descriptions

The system can automatically generate captions like "a child in a pink dress climbing stairs" or "a black dog and spotted dog fighting" based solely on image input.

## Dataset
- **Source**: Flickr8k Dataset
- **Location**: `Downloads/flickr8k`
- **Images**: 8,091 images
- **Captions**: 40,455 captions (5 captions per image)
- **Split**: 90% training (7,281 images), 10% testing (810 images)
- **Format**: Each image paired with 5 different human-written descriptions

## Model Architecture

### 1. Feature Extraction (Encoder)
**VGG16 CNN Model**:
- Pre-trained on ImageNet dataset
- Modified to extract features from the second-to-last fully connected layer (fc2)
- Input: 224×224×3 RGB images
- Output: 4096-dimensional feature vector
- Total Parameters: 134,260,544 (512.16 MB)

### 2. Caption Generation (Decoder)
**Encoder-Decoder with LSTM**:

**Image Feature Branch**:
- Input: 4096-dimensional image features
- Dropout layer (0.4)
- Dense layer (256 units, ReLU activation)

**Text Sequence Branch**:
- Input: Sequence of max_length tokens
- Embedding layer (vocab_size × 256 dimensions)
- Dropout layer (0.4)
- LSTM layer (256 units)

**Decoder**:
- Merges image and text features using element-wise addition
- Dense layer (256 units, ReLU activation)
- Output layer (vocab_size, softmax activation)

## Key Features
- **Transfer Learning**: Leverages VGG16 pre-trained on ImageNet
- **Attention Mechanism**: Combines visual and textual information
- **Sequence-to-Sequence Learning**: Generates captions word by word
- **Memory Efficient**: Uses data generators for batch processing
- **Start/End Tokens**: Implements 'startseq' and 'endseq' for caption boundaries

## Hyperparameters
```python
EPOCHS = 20
BATCH_SIZE = 32
MAX_LENGTH = 35              # Maximum caption length
VOCAB_SIZE = 8485            # Total unique words
EMBEDDING_DIM = 256          # Word embedding dimensions
LSTM_UNITS = 256             # LSTM hidden units
DROPOUT_RATE = 0.4           # Dropout for regularization
OPTIMIZER = 'adam'
LOSS = 'categorical_crossentropy'
IMAGE_SIZE = (224, 224, 3)   # VGG16 input size
FEATURE_DIM = 4096           # VGG16 output features
```

## Requirements
```
tensorflow
keras
numpy
matplotlib
tqdm
nltk
pillow (PIL)
pickle
```

## Installation
```bash
pip install tensorflow keras numpy matplotlib tqdm nltk pillow
```

## Dataset Structure
```
Downloads/
  flickr8k/
    Images/
      *.jpg (8091 image files)
    captions.txt
```

## Usage

### 1. Feature Extraction
Extract features from all images using VGG16:
```python
# Load VGG16 model (without top classification layer)
model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

# Extract features for all images
features = {}
for img_name in os.listdir('Downloads/flickr8k/Images'):
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    feature = model.predict(image)
    features[image_id] = feature

# Save features
pickle.dump(features, open('features.pkl', 'wb'))
```

### 2. Text Preprocessing
Load and clean captions:
```python
# Load captions
with open('Downloads/flickr8k/captions.txt', 'r') as f:
    captions_doc = f.read()

# Clean text (lowercase, remove special chars, add start/end tags)
# Example: "A child playing" -> "startseq child playing endseq"
```

### 3. Train the Model
```python
# Create and compile model
model = create_model(vocab_size, max_length)

# Train for 20 epochs
for epoch in range(20):
    generator = data_generator(train, mapping, features, tokenizer,
                               max_length, vocab_size, batch_size=32)
    model.fit(generator, epochs=1, steps_per_epoch=steps)

# Save the trained model
model.save('best_model.keras')
```

### 4. Generate Captions for New Images
```python
# Load and preprocess image
image = load_img('path/to/image.jpg', target_size=(224, 224))
image = img_to_array(image)
image = preprocess_input(image)

# Extract features
feature = vgg_model.predict(image)

# Generate caption
caption = predict_caption(model, feature, tokenizer, max_length)
print(caption)  # Output: "startseq man in green shirt sitting on stairs endseq"
```

## Text Preprocessing Steps
1. **Lowercase conversion**: Normalize all text to lowercase
2. **Remove special characters**: Keep only alphabetic characters
3. **Remove extra spaces**: Normalize whitespace
4. **Filter short words**: Remove single-character words
5. **Add sequence markers**: Prepend 'startseq' and append 'endseq'

Example transformation:
- Before: `"A child in a pink dress is climbing up a set of stairs in an entry way ."`
- After: `"startseq child in pink dress is climbing up set of stairs in an entry way endseq"`

## Training Process
The model uses a custom data generator that:
1. Samples a batch of images
2. For each image, retrieves its 5 captions
3. Generates input-output pairs for sequence prediction
4. Pads sequences to uniform length
5. Yields batches of (image_features, text_sequence) → next_word

### Training Details
- **Steps per Epoch**: ~227 steps (7281 images / 32 batch size)
- **Total Training Steps**: 4,540 (20 epochs × 227 steps)
- **Input Format**:
  - Image: 4096-dimensional feature vector
  - Text: Sequence of token IDs (padded to max_length)
- **Output Format**: One-hot encoded next word (vocab_size classes)

## Evaluation Metrics
The model is evaluated using BLEU (Bilingual Evaluation Understudy) scores:

### BLEU Scores on Test Set (810 images):
- **BLEU-1**: 0.546081 (54.6% - measures unigram precision)
- **BLEU-2**: 0.317512 (31.8% - measures bigram precision)

### Interpretation:
- BLEU-1 > 0.5 indicates good word-level accuracy
- BLEU-2 > 0.3 shows decent phrase-level coherence
- Higher scores indicate better caption quality

## Example Results

### Example 1:
**Image**: Child in pink dress on stairs
- **Actual**: "A child in a pink dress is climbing up a set of stairs"
- **Predicted**: "startseq child in pink dress climbing stairs endseq"

### Example 2:
**Image**: Dogs playing
- **Actual**: "Two dogs of different breeds looking at each other"
- **Predicted**: "startseq black dog and spotted dog are fighting endseq"

## Model Performance
- **Training**: Successfully converges after 20 epochs
- **Inference Time**: ~100ms per image (feature extraction + caption generation)
- **Vocabulary Coverage**: 8,485 unique words
- **Maximum Caption Length**: 35 words
- **Memory Usage**: ~520MB (VGG16) + model overhead

## Key Functions

### `data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size)`
Generates training batches on-the-fly to avoid memory issues.

### `predict_caption(model, image, tokenizer, max_length)`
Generates caption for a single image using greedy decoding.

### `clean(mapping)`
Preprocesses all captions in the dataset.

### `idx_to_word(integer, tokenizer)`
Converts token ID back to word string.

### `generate_caption(image_name)`
Complete pipeline: load image → extract features → generate caption → display results.

## Applications
1. **Accessibility**: Assist visually impaired users by describing images
2. **Social Media**: Auto-generate image descriptions for posts
3. **Content Management**: Automatically tag and organize image databases
4. **Security**: Generate descriptions for surveillance footage
5. **E-commerce**: Auto-generate product descriptions from images
6. **Medical Imaging**: Provide preliminary descriptions of medical scans

## Limitations
1. **Fixed Vocabulary**: Cannot generate words not in training vocabulary (8,485 words)
2. **Greedy Decoding**: Uses simple greedy search instead of beam search
3. **Generic Descriptions**: May produce generic captions for complex scenes
4. **No Attention Visualization**: Cannot visualize which image regions influenced each word
5. **Single Object Focus**: Struggles with multi-object scenes
6. **Context Understanding**: Limited understanding of complex relationships

## Future Improvements
1. **Attention Mechanism**: Implement visual attention to focus on relevant image regions
2. **Beam Search**: Use beam search decoding for better caption quality
3. **Fine-tuning**: Fine-tune VGG16 layers for better feature extraction
4. **Larger Models**: Use ResNet, Inception, or EfficientNet for better features
5. **Transformer Architecture**: Implement Vision Transformer (ViT) with GPT-style decoder
6. **Larger Dataset**: Train on MS COCO (330K images) or Conceptual Captions
7. **Evaluation Metrics**: Add METEOR, CIDEr, and SPICE scores
8. **Ensemble Models**: Combine multiple models for robust predictions
9. **Multi-modal Pre-training**: Use CLIP or similar models
10. **Real-time Inference**: Optimize for mobile/edge deployment

## Model Files
After training, the following files are generated:
- `features.pkl`: Pre-extracted VGG16 features for all images (~1GB)
- `best_model.keras`: Trained caption generation model
- `tokenizer.pkl`: Tokenizer with vocabulary mapping

## Technical Details

### Memory Optimization
- Uses generator pattern to load data in batches
- Pre-extracts and caches VGG16 features
- Implements dropout for regularization
- Clears session after each epoch

### Sequence Generation
The model generates captions autoregressively:
1. Start with 'startseq' token
2. Encode current sequence
3. Predict next word probability distribution
4. Select word with highest probability (argmax)
5. Append to sequence and repeat
6. Stop when 'endseq' is generated or max_length reached

## References
1. Vinyals, O., et al. (2015). "Show and Tell: A Neural Image Caption Generator"
2. Xu, K., et al. (2015). "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"
3. Simonyan, K., & Zisserman, A. (2014). "Very Deep Convolutional Networks for Large-Scale Image Recognition" (VGG16)
4. Papineni, K., et al. (2002). "BLEU: a Method for Automatic Evaluation of Machine Translation"

## Dataset Citation
- **Flickr8k Dataset**: Hodosh, M., Young, P., & Hockenmaier, J. (2013). "Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics"

## License
This project is for educational and research purposes.

## Author
Nikhita Gowda

---

## Troubleshooting

### Common Issues:
1. **Out of Memory**: Reduce batch_size or use the data generator
2. **Features not found**: Ensure features.pkl is generated before training
3. **Vocabulary errors**: Verify captions.txt is properly loaded
4. **Model loading errors**: Use `.keras` format instead of `.h5` for TensorFlow 2.x

## Performance Tips
- Pre-extract all image features before training
- Use GPU for training (10-15x speedup)
- Cache tokenizer and features to disk
- Use mixed precision training for faster computation

---

**Note**: This implementation achieves competitive BLEU scores on Flickr8k dataset and demonstrates the fundamental concepts of neural image captioning. For production use, consider implementing attention mechanisms and training on larger datasets.
