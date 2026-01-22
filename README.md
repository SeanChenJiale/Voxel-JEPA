# NAASSL

![NAASSL framework](updated_framework.png)
NAASSL is a deep learning framework designed for analyzing and processing 3D medical imaging data. It includes utilities for entropy-based subcortical analysis, video reconstruction, and correlation-based metrics for neuroimaging datasets. 
The repository leverages PyTorch and ANTs for efficient tensor operations and image processing.

---

## Features

- **Entropy Analysis**: Extract entropy and statistical measures (mean, standard deviation) for subcortical regions in 3D medical images.
- **Correlation Metrics**: Compute correlation matrices and traces for subcortical regions across multiple samples.
- **Video Reconstruction**: Reshape and preprocess video data for downstream tasks.
- **Customizable Analysis**: Easily switch between different statistical measures (e.g., mean, standard deviation) for subcortical analysis.

---

## Repository Structure

NAASSL/ 

├── jepa/

│ ├── src/ 

│ │ ├── utils/ 

│ │ │ ├── entropy_loss.py # Core utility for entropy and trace calculations 

│ │ ├── models/ # Model definitions 

│ │ ├── configs/ # Configuration files for experiments 

│ ├── wandb/ # Weights & Biases logging 

├── README.md # Project documentation

├── requirements.txt # Python dependencies 

└── LICENSE # License information


---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SeanChenJiale/NAASSL.git
   cd NAASSL
   ```
2. Create a Python virtual environment:
   ```bash
   python -m venv naassl
   source naassl/bin/activate  # On Windows: naassl\Scripts\activate
   ```
3. Install the required dependancies:
   ```bash
   pip install -r requirements.txt
   ```

Usage
1. Subcortical Entropy Analysis
The entropy_loss.py script provides functions to compute entropy and statistical measures for subcortical regions in 3D medical images.

``` python 
from jepa.src.utils.entropy_loss import calculate_trace_torch

# Inputs
batch_reconstructed_video = [...]  # List of tensors representing video data
batch_indices_list = [...]         # List of slice indices for each sample
batch_axis_list = [...]            # List of axes for each sample
atlas_list = [...]                 # List of atlases for subcortical regions

# Compute trace
trace = calculate_trace_torch(
    batch_reconstructed_video,
    batch_indices_list,
    batch_axis_list,
    atlas_list=atlas_list,
    trace_type='mean'
)
print(trace)
```

Acknowledgements
This project was developed for advanced neuroimaging analysis and leverages tools like PyTorch and ANTs for efficient computation the underlying framework is taken from Meta, V-JEPA.
Special thanks to the open-source community for providing the foundational libraries used in this project.
