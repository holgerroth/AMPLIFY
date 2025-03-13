import os
import amplify
import torch

#model_dir = "/local/path/to/model"
model_dir = "/localhome/local-hroth/Data/AMPLIFY/AMPLIFY_120M"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
config_path = os.path.join(model_dir, "config.yaml")
checkpoint_file = os.path.join(model_dir, "pytorch_model.pt")

model, tokenizer = amplify.AMPLIFY.load(checkpoint_file, config_path)

# Link the model to the inference API:
predictor = amplify.inference.Predictor(model, tokenizer, device=device)

# Calculate logits for a sequence
sequence = "MSVVGIDLGFQSCYVAVARAGGIETIANEYSDRCTPACISFGPKNR"
logits = predictor.logits(sequence)
print("sequence:", len(sequence) ,"logits:", logits.shape)

# Get the embedding for a given sequence:
embedder = (model, tokenizer)
sequence_embedding = predictor.embed(sequence)
print("sequence:", len(sequence), "sequence_embedding:", sequence_embedding.shape)

# Compare the sequence to several other sequences
other_sequences = [
    "AACGGEVWVTDEAAAAA",
    "AAAAACGGGVWWTDEAAAAA",
    "AAAADGGVWVTECDA",
]

other_sequence_embeddings = [predictor.embed(x) for x in other_sequences]
print("other_sequence_embeddings:")
[print("sequence:", len(s), "sequence_embedding", e.shape) for s, e in zip(other_sequences, other_sequence_embeddings)]

import numpy as np
other_embedding_mean = np.mean(np.concatenate(other_sequence_embeddings), axis=0)
print("other_embedding_mean", other_embedding_mean.shape)
similarities = amplify.inference.cosine_similarities(
    other_embedding_mean,
    sequence_embedding,
)
print("similarities:", similarities)
