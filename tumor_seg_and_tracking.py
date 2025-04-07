import logging
from pathlib import Path
import traceback
import torch
import os
import SimpleITK as sitk
from convexAdam.convex_adam_MIND import convex_adam_pt
import numpy as np
from convexAdam.convex_adam_utils import (resample_img,
                                          resample_moving_to_fixed)
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

# Configure logging at the beginning of the file
def setup_logging(log_file='registration.log'):
    """
    Configure the logging system.
    
    Args:
        log_file (str, optional): Path to the log file. Defaults to 'registration.log'.
    
    Returns:
        logger: Configured logger instance.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger

def configure_torch_settings():
    """
    Configure PyTorch's deterministic settings and initialize inference session.
    
    Returns:
        nnInteractiveInferenceSession: Initialized inference session
    """
    # Configure deterministic settings for reproducibility
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Initialize inference session with optimal settings
    session = nnInteractiveInferenceSession(
        device=torch.device("cuda:0"),
        use_torch_compile=False,  # Experimental feature
        verbose=False,
        torch_n_threads=os.cpu_count(),
        do_autozoom=True,
        use_pinned_memory=True,
    )
    
    model_path =  "nnInteractive_v1.0"
    session.initialize_from_trained_model_folder(model_path)
    return session

def load_dicom_series(dicom_dir, series_name):
    """
    Load DICOM series from specified directory.
    
    Args:
        dicom_dir (str): Root directory containing DICOM series
        series_name (str): Name of the DICOM series to load
    
    Returns:
        sitk.Image: Loaded DICOM series as SimpleITK image
    """
    reader = sitk.ImageSeriesReader()
    dicom_files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
        os.path.join(dicom_dir, series_name)
    )
    reader.SetFileNames(dicom_files)
    return reader.Execute()

def perform_registration(fixed_img, moving_img):
    """
    Perform image registration using large grid spacing configuration.
    
    Args:
        fixed_img (sitk.Image): Reference/fixed image
        moving_img (sitk.Image): Image to be aligned
    
    Returns:
        np.ndarray: Displacement field from registration
    """
    return convex_adam_pt(
        img_fixed=fixed_img,
        img_moving=moving_img,
        dtype=torch.float64,
        selected_niter=160,   # Reduced initial iterations
        mind_d=8,             # Increased feature distance
        mind_r=4,             # Increased feature radius
        disp_hw=16,           # Expanded displacement search range
        grid_sp=12,           # Larger grid spacing
        lambda_weight=0.5,    # Reduced deformation field constraint
        selected_smooth=5,    # Enhanced smoothing
        ic=True,
    )

def calculate_tumor_position(fixed_image, displacement_field, scale_factor, 
                            input_point, patch_size, return_physical_point):
    """
    Calculate tumor position after deformation.
    
    Args:
        fixed_image (sitk.Image): Reference image for coordinate system
        displacement_field (np.ndarray): Deformation field data
        scale_factor (list): Scaling factors for each dimension
        input_point (tuple): Initial tumor position (z,y,x)
        patch_size (tuple): Size of the deformation patch (z,y,x)
        return_physical_point (bool): Return physical coordinates if True
    
    Returns:
        list: New tumor position in either voxel or physical coordinates
    """
    z1, y1, x1 = input_point
    half_z, half_y, half_x = [dim//2 for dim in patch_size]

    # Extract displacement values from deformation field
    dx = displacement_field[half_z, half_y, half_x, 2] / scale_factor[2]
    dy = displacement_field[half_z, half_y, half_x, 1] / scale_factor[1]
    dz = displacement_field[half_z, half_y, half_x, 0] / scale_factor[0]

    new_voxel = [round(z1+dz), round(y1+dy), round(x1+dx)]
    
    if return_physical_point:
        physical_point = fixed_image.TransformIndexToPhysicalPoint(
            (new_voxel[2], new_voxel[1], new_voxel[0]))
        return [physical_point[2], physical_point[1], physical_point[0]]
    return new_voxel

def input_point_cal(img, scale_factor, input_point, input_physical_point):
    """
    Calculate coordinates in original and resampled spaces.
    
    Args:
        img (sitk.Image): Reference image
        scale_factor (list): Resampling scale factors
        input_point (tuple): Initial point coordinates
        input_physical_point (bool): Whether input is in physical coordinates
    
    Returns:
        tuple: (original_coordinates, resampled_coordinates)
    """
    if input_physical_point:
        index = img.TransformPhysicalPointToIndex(
            (input_point[2], input_point[1], input_point[0]))
        original_center = [index[2], index[1], index[0]]
    else:
        original_center = input_point

    resample_center = [
        round(original_center[0] * scale_factor[0]),
        round(original_center[1] * scale_factor[1]),
        round(original_center[2] * scale_factor[2]),
    ]
    return original_center, resample_center

def get_patch(sitk_image, resample_center, patch_size):
    """
    Extract image patch with proper padding handling.
    
    Args:
        sitk_image (sitk.Image): Input image
        resample_center (tuple): Center point for patch extraction
        patch_size (tuple): Desired patch dimensions
    
    Returns:
        sitk.Image: Extracted image patch
    """
    arr = sitk.GetArrayFromImage(sitk_image)
    z, y, x = resample_center

    # Calculate crop boundaries with padding
    start_z = z - patch_size[0]//2
    end_z = start_z + patch_size[0]
    start_y = y - patch_size[1]//2
    end_y = start_y + patch_size[1]
    start_x = x - patch_size[2]//2
    end_x = start_x + patch_size[2]

    # Initialize padding values
    pad_z = [0, 0]
    pad_y = [0, 0]
    pad_x = [0, 0]

    # Handle z-axis boundaries
    if start_z < 0:
        pad_z[0] = abs(start_z)
        start_z = 0
    if end_z > arr.shape[0]:
        pad_z[1] = end_z - arr.shape[0]
        end_z = arr.shape[0]

    # Handle y-axis boundaries
    if start_y < 0:
        pad_y[0] = abs(start_y)
        start_y = 0
    if end_y > arr.shape[1]:
        pad_y[1] = end_y - arr.shape[1]
        end_y = arr.shape[1]

    # Handle x-axis boundaries
    if start_x < 0:
        pad_x[0] = abs(start_x)
        start_x = 0
    if end_x > arr.shape[2]:
        pad_x[1] = end_x - arr.shape[2]
        end_x = arr.shape[2]

    # Create and pad output array
    cropped = np.pad(
        arr[start_z:end_z, start_y:end_y, start_x:end_x],
        pad_width=(pad_z, pad_y, pad_x),
        mode='constant'
    )
    
    patch = sitk.GetImageFromArray(cropped)
    patch.CopyInformation(sitk_image)
    return patch

def pred_mask(session, image, point):
    """
    Generate segmentation mask using interactive model.
    
    Args:
        session: Initialized inference session
        image (sitk.Image): Input image
        point (tuple): Seed point coordinates (z,y,x)
    
    Returns:
        np.ndarray: Generated segmentation mask
    """
    arr = sitk.GetArrayFromImage(image)
    session.set_image(np.expand_dims(arr, 0))
    
    target = torch.zeros(arr.shape, dtype=torch.int16)
    session.set_target_buffer(target)
    session.add_point_interaction(tuple(point), include_interaction=True)
    
    mask = session.target_buffer.clone().cpu().numpy().squeeze()
    session.reset_interactions()
    return mask.astype(np.int16)

def dice_score(y_true, y_pred):
    """
    Calculate Dice similarity coefficient.
    
    Args:
        y_true (np.ndarray): Ground truth binary mask
        y_pred (np.ndarray): Predicted binary mask
    
    Returns:
        float: Dice score between 0 and 1
    """
    smooth = 1e-6
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

def main(config):
    """
    Main registration pipeline execution.
    
    Args:
        config (dict): Configuration parameters
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    try:
        logger.info("Starting registration pipeline")
        logger.debug(f"Input config: {config}")

        # Validate input directory
        if not Path(config["input_dir"]).exists():
            raise FileNotFoundError(f"Input directory not found: {config['input_dir']}")

        # Load imaging data
        try:
            logger.info("Loading fixed image...")
            fixed_image = load_dicom_series(config["input_dir"], config["fixed_series"])
            logger.info(f"Fixed image loaded - Size: {fixed_image.GetSize()}")
            
            logger.info("Loading moving image...")
            moving_image = load_dicom_series(config["input_dir"], config["moving_series"])
            logger.info(f"Moving image loaded - Size: {moving_image.GetSize()}")
        except Exception as e:
            logger.error("Image loading failed", exc_info=True)
            raise

        # Initialize computational framework
        try:
            logger.info("Initializing PyTorch...")
            session = configure_torch_settings()
            logger.info("PyTorch initialized successfully")
        except Exception as e:
            logger.error("PyTorch initialization failed", exc_info=True)
            raise

        # Calculate spatial transformations
        try:
            logger.debug("Calculating scaling factors...")
            scale_factor = [
                fixed_image.GetSpacing()[2] / config["resample_spacing"][0],
                fixed_image.GetSpacing()[1] / config["resample_spacing"][1],
                fixed_image.GetSpacing()[0] / config["resample_spacing"][2]
            ]
            logger.debug(f"Scale factors: {scale_factor}")
        except Exception as e:
            logger.error("Scale calculation error", exc_info=True)
            raise

        # Image preprocessing
        try:
            logger.info("Resampling images...")
            fixed_resampled = resample_img(fixed_image, config["resample_spacing"])
            moving_resampled = resample_moving_to_fixed(fixed_resampled, moving_image)
            logger.info(f"Resampled sizes | Fixed: {fixed_resampled.GetSize()}, Moving: {moving_resampled.GetSize()}")
        except Exception as e:
            logger.error("Resampling failed", exc_info=True)
            raise

        # Region of Interest extraction
        try:
            logger.info("Extracting ROIs...")
            orig_center, resamp_center = input_point_cal(
                fixed_image, scale_factor,
                config["fixed_point"], config["input_physical_point"]
            )
            fixed_patch = get_patch(fixed_resampled, resamp_center, config["patch_size"])
            moving_patch = get_patch(moving_resampled, resamp_center, config["patch_size"])
            logger.info(f"ROIs extracted - Size: {config['patch_size']}")
        except Exception as e:
            logger.error("ROI extraction failed", exc_info=True)
            raise

        # Core registration
        try:
            logger.info("Performing registration...")
            displacement = perform_registration(fixed_patch, moving_patch)
            logger.info("Registration completed successfully")
        except Exception as e:
            logger.error("Registration failed", exc_info=True)
            raise

        # Post-registration analysis
        try:
            logger.info("Calculating tumor displacement...")
            new_position = calculate_tumor_position(
                fixed_image,
                displacement,
                scale_factor,
                orig_center,
                config["patch_size"],
                config["return_physical_point"]
            )
            logger.info(f"Original position: {config['fixed_point']}")
            logger.info(f"New position: {new_position}")
        except Exception as e:
            logger.error("Position calculation error", exc_info=True)
            raise

        # Quality assurance
        try:
            logger.info("Validating results...")
            validation_point = [
                round(new_position[0] * scale_factor[0]),
                round(new_position[1] * scale_factor[1]),
                round(new_position[2] * scale_factor[2])
            ]
            pred_patch = get_patch(moving_resampled, validation_point, config["patch_size"])
            
            fixed_mask = pred_mask(session, fixed_patch, [s//2 for s in config["patch_size"]])
            moving_mask = pred_mask(session, moving_patch, [s//2 for s in config["patch_size"]])
            pred_mask = pred_mask(session, pred_patch, [s//2 for s in config["patch_size"]])
            
            dice_original = dice_score(fixed_mask, moving_mask)
            dice_registered = dice_score(fixed_mask, pred_mask)
            logger.info(f"Dice Scores | Original: {dice_original:.3f}, Registered: {dice_registered:.3f}")
            
            if dice_registered > dice_original and dice_registered > 0.65:
                logger.info("Validation successful")
            else:
                logger.warning("Quality threshold not met")
        except Exception as e:
            logger.error("Validation failed", exc_info=True)
            raise

    except Exception as e:
        logger.error(f"Pipeline aborted: {str(e)}")
        logger.debug(traceback.format_exc())
        return False
    finally:
        logger.info("Pipeline completed")
    return True

if __name__ == "__main__":
    # Configuration parameters
    config = {
        "input_dir": r"D:\project\data\intraoperative_MRI\6\sorted_new",
        "fixed_series": r"201000\0",
        "moving_series": r"301000\0",
        "fixed_point": [22, 256, 195],  # z,y,x order
        "input_physical_point": False,
        "return_physical_point": False,
        "patch_size": [64, 64, 64],     # z,y,x dimensions
        "resample_spacing": [1, 1, 1],  # z,y,x in mm
    }
    
    # Initialize logging
    logger = setup_logging()
    # Execute main pipeline
    try:
        success = main(config)
        if not success:
            exit(1)
    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user")
        exit(130)
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}", exc_info=True)
        exit(2)