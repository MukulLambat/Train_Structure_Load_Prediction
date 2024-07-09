import torch

# Check for GPU availability
use_cuda = torch.cuda.is_available()

if use_cuda:
  print("GPU is available!")
  # Get additional information (optional)
  print(f"Number of CUDA devices: {torch.cuda.device_count()}")
else:
  print("No GPU available.")