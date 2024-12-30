import torch
import torch.nn as nn
from model_mamba import Mamba_model
from diffustion_transformer import Dit

class Mad(nn.Module):
    def __init__(self, shape_in_reconstruction, shape_in_prediction, scaling_coeff=0.1, device=None):
        """
        Initializes the Mad model by setting up the reconstruction and prediction models.

        Args:
            shape_in_reconstruction (tuple): Input shape for the reconstruction model (e.g., (10, 1, 64, 64)).
            shape_in_prediction (tuple): Input shape for the prediction model (e.g., (1, 10, 1, 64, 64)).
            scaling_coeff (float, optional): Coefficient to scale the reconstruction features before addition. Defaults to 1.0.
            device (torch.device, optional): Device to run the models on. If None, defaults to CUDA if available.
        """
        super(Mad, self).__init__()
        
        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Initialize models
        self.model_reconstruction = Mamba_model(shape_in=shape_in_reconstruction).to(self.device)
        self.model_prediction = Dit(shape_in=shape_in_prediction).to(self.device)
        
        # Scaling coefficient
        self.scaling_coeff = scaling_coeff

    def forward(self, inputs):
        """
        Forward pass that processes the inputs through both models and combines their outputs.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Combined output tensor.
        """
        # Move inputs to the correct device
        inputs = inputs.to(self.device)
        
        # Get features from both models
        rec_feature = self.model_reconstruction(inputs)
        pred_feature = self.model_prediction(inputs)
        
        # Ensure both features have the same shape for addition
        if rec_feature.shape != pred_feature.shape:
            raise ValueError(f"Shape mismatch: rec_feature shape {rec_feature.shape} vs pred_feature shape {pred_feature.shape}")
        
        # Combine features with scaling coefficient
        combined = self.scaling_coeff * rec_feature + pred_feature
        
        return combined

# Example usage
if __name__ == "__main__":
    inputs = torch.randn(1, 10, 1, 64, 64)  # Example input
    
    mad_model = Mad(
        shape_in_reconstruction=(10, 1, 64, 64),
        shape_in_prediction=(1, 10, 1, 64, 64),
        scaling_coeff=0.1
    )
    
    # Forward pass
    output = mad_model(inputs)
    
    # Print output shape
    print("Combined Output shape:", output.shape) # 1, 10, 1, 64, 64
