# GMM Optimal Transport (GMM_OT)

This repository implements Gaussian Mixture Model (GMM) Optimal Transport for mapping samples between two GMM-distributed datasets. It includes utilities for computing optimal couplings, transporting samples, and visualizing the results.

The experiments and theory are based on the paper https://hal.archives-ouvertes.fr/hal-02178204.

## Repository Structure

- `gmmot.py`: Contains the main implementation of GMM Optimal Transport functions.
- `utils.py`: Contains utility functions for visualizing some of the results and loading some datasets.
- `models.py`: Contains the implementation of the AE and VAE models used in the experiments.
- `GMM_OT.ipynb`: A Jupyter notebook containing our experiments and visualizations of GMM Optimal Transport.

## Setup

1. Clone this repository:

```bash
git clone https://github.com/samsongourevitch/GMM_OT.git
cd GMM_OT
```

2. Create a virtual environment and install dependencies:

```bash
python3 -m venv gmm_ot_env
source gmm_ot_env/bin/activate
pip install -r requirements.txt
```

3. Download the MNIST dataset and place it in the root directory of the repository:

```python
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
train_set = MNIST(MNIST_path, train=True, transform=transform, download=True)
``` 

If you run the code again, just set `download=False`.



