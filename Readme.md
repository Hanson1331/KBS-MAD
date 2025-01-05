~~~markdown
# MAD: Mamba + Diffusion Transformer for Spatiotemporal Prediction

This project provides a spatiotemporal prediction framework that combines **Mamba** modules with **Diffusion Transformers** (MAD). It applies to tasks such as fluid simulation, weather forecasting, and traffic flow analysis.

## Features

- **Mamba Module**: Uses bidirectional convolution and state-space modeling to capture spatiotemporal dependencies and reduce error propagation.
- **Diffusion Transformer**: Employs self-attention and diffusion mechanisms to handle long-range spatiotemporal correlations and generate high-quality predictions.
- **Joint Training**: Optimizes both reconstruction and prediction tasks to learn more representative spatiotemporal features.
- **Scalability**: Adapts to various datasets by using custom data loaders.

## Directory Structure

- `mamba_dit/`
  - **Core Code**: `mad_model.py`, `diffustion_transformer.py`, `modules.py`, etc.
  - **Data Loading**: `api_dataloader/dataloader_ns.py`
  - **Training Script**: `train_model.py`
  - **Custom Layers**: `layers/`
  - **Logs & Checkpoints**: `logs/`, `checkpoints/`
  - **Visualization & Testing**: `plt.ipynb`, `test_model.ipynb`

## Quick Start

1. **Environment Setup**  
   - Python 3.8+  
   - PyTorch (install the CUDA version that matches your hardware)  
   - Other dependencies listed in `requirements.txt`  
   ```bash
   conda create -n mad_env python=3.8
   conda activate mad_env
   conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
   pip install -r requirements.txt
~~~

1. **Data Preparation**

   - Place your data in a specified directory (e.g., `data/navier_stokes/`).
   - Modify paths in `dataloader_ns.py` to match your setup.

2. **Train the Model**

   ```bash
   python mamba_dit/train_model.py \
       --dataset navier_stokes \
       --batch_size 16 \
       --epochs 100 \
       --lr 1e-4 \
       --save_dir ./mamba_dit/checkpoints
   ```

   - Parameters

     :

     - `--dataset`: Dataset name (customizable)
     - `--batch_size`: Training batch size
     - `--epochs`: Number of training epochs
     - `--lr`: Initial learning rate
     - `--save_dir`: Directory for saving model checkpoints

3. **Testing & Visualization**

   - Open `test_model.ipynb` or `plt.ipynb`, set the path for the pre-trained weights and dataset, then run the cells to view prediction outcomes and plots.

## Citation

If you find this project helpful, please cite:

```bibtex
@article{Zeng2025MAD,
  title={Enhancing Spatiotemporal Prediction through the Integration of Mamba State Space Models and Diffusion Transformers},
  author={Zeng, Hansheng and Li, Yuqi and Niu Ruize and Yang, Chuanguang and Wen, Shiping},
  journal={Knowledge based system},
  year={2025},
}
```

## License

This project is released under the MIT License. See [LICENSE](https://chatgpt.com/c/LICENSE) for details.

For questions or suggestions, please open an Issue or contact us via email.

```

```