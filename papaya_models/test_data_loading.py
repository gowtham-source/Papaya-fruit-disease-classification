import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('test_data_loading')

def check_data_paths():
    # Get the current working directory
    cwd = os.getcwd()
    logger.info(f"Current working directory: {cwd}")
    
    # Define expected data paths
    data_dir = os.path.join(cwd, 'data')
    train_dir = os.path.join(data_dir, 'Train')
    
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Train directory: {train_dir}")
    
    # Check if directories exist
    logger.info(f"Data directory exists: {os.path.exists(data_dir)}")
    logger.info(f"Train directory exists: {os.path.exists(train_dir)}")
    
    if os.path.exists(train_dir):
        # List some files in the train directory
        files = os.listdir(train_dir)
        logger.info(f"Found {len(files)} files in train directory")
        
        # Count images and annotations
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in files if os.path.splitext(f)[1].lower() in image_exts]
        txt_files = [f for f in files if f.lower().endswith('.txt')]
        
        logger.info(f"Found {len(image_files)} image files")
        logger.info(f"Found {len(txt_files)} annotation files")
        
        # Check for matching pairs
        if image_files and txt_files:
            base_names = {os.path.splitext(f)[0] for f in image_files}
            txt_base_names = {os.path.splitext(f)[0] for f in txt_files}
            matches = base_names.intersection(txt_base_names)
            logger.info(f"Found {len(matches)} matching image-annotation pairs")
            
            if matches:
                sample = list(matches)[0]
                logger.info(f"Sample match: {sample}.jpg/.txt")
        
        # Print first few files
        logger.info("Sample files in train directory:")
        for f in files[:5]:
            logger.info(f"  - {f}")
    else:
        logger.error(f"Train directory not found at: {train_dir}")
        # List contents of data directory for debugging
        if os.path.exists(data_dir):
            logger.info(f"Contents of data directory: {os.listdir(data_dir)}")

if __name__ == "__main__":
    check_data_paths()
