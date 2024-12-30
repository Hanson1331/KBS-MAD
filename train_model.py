import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import scipy.io
from api_dataloader.dataloader_ns import NSDataset 
from mad_model import Mad  
import numpy as np
from tqdm import tqdm  
import logging  

# Set up logging
LOG_FILE = 'log.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration parameters
DATA_PATH = '/Users/wuhao/Desktop/mamba_dit/dataset/NavierStokes_V1e-5_N1200_T20.mat'
BATCH_SIZE = 4
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
SCALING_COEFF = 0.1
CHECKPOINT_DIR = '/Users/wuhao/Desktop/mamba_dit/checkpoints'
LOG_DIR = '/Users/wuhao/Desktop/mamba_dit/logs'
RESULTS_DIR = '/Users/wuhao/Desktop/mamba_dit/results'

# Create directories if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Initialize TensorBoard writer
writer = SummaryWriter(LOG_DIR)

# Data loaders
def get_dataloaders(path_file, batch_size=1, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    dataset = NSDataset(path_file=path_file)
    total_size = len(dataset)
    
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # Ensure consistent splits
    torch.manual_seed(42)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = get_dataloaders(
    path_file=DATA_PATH,
    batch_size=BATCH_SIZE
)

# Initialize model
mad_model = Mad(
    shape_in_reconstruction=(10, 1, 64, 64),
    shape_in_prediction=(1, 10, 1, 64, 64),
    scaling_coeff=SCALING_COEFF
)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mad_model.to(device)
logger.info(f'Model moved to device: {device}')

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(mad_model.parameters(), lr=LEARNING_RATE)
logger.info('Loss function and optimizer initialized.')

# Training function
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if (batch_idx + 1) % 100 == 0:
            logger.info(f'Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
    
    return running_loss / len(dataloader)

# Validation function
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validation", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
    
    return running_loss / len(dataloader)

# Testing function
def test_model(model, dataloader, device, save_path):
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Testing", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Save results as NPZ
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    np.savez(save_path, predictions=all_predictions, targets=all_targets)
    logger.info(f"Test results saved to {save_path}")

# Training and validation loop
best_val_loss = float('inf')
for epoch in tqdm(range(1, NUM_EPOCHS + 1), desc="Epochs"):
    logger.info(f'Epoch [{epoch}/{NUM_EPOCHS}]')
    
    train_loss = train_epoch(mad_model, train_loader, criterion, optimizer, device)
    val_loss = validate_epoch(mad_model, val_loader, criterion, device)
    
    logger.info(f'Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
    
    # Log to TensorBoard
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Loss/Validation', val_loss, epoch)
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'mad_model_epoch_{epoch}.pth')
        torch.save(mad_model.state_dict(), checkpoint_path)
        logger.info(f'Best model saved to {checkpoint_path}')
    
    # Save checkpoint every 10 epochs
    if epoch % 10 == 0:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'mad_model_epoch_{epoch}.pth')
        torch.save(mad_model.state_dict(), checkpoint_path)
        logger.info(f'Model checkpoint saved to {checkpoint_path}')

# Close TensorBoard writer
writer.close()
logger.info('TensorBoard writer closed.')

# Testing phase
test_checkpoint = os.path.join(CHECKPOINT_DIR, f'mad_model_epoch_{epoch}.pth')
mad_model.load_state_dict(torch.load(test_checkpoint))
test_save_path = os.path.join(RESULTS_DIR, 'test_results.npz')
test_model(mad_model, test_loader, device, test_save_path)

logger.info("Training and testing completed.")
