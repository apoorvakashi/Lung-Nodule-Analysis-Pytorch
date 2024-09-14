# LUNG NODULE ANALYSIS - PYTORCH 
<img width="753" alt="Screen Shot 2024-09-05 at 5 39 16 PM" src="https://github.com/user-attachments/assets/4bc2d536-c997-4450-b8f8-47596d335f65">

# MODULES
  1. [Combining Data](#combining-data)
  2. [Classification](#classification)
  3. [Segmentation](#segmentation)

____________________________________________________________________________________________________

## Combining Data

The raw CT scan data has to be loaded into a form that can be used with PyTorch. The CT data comes in two files: a .mhd file containing metadata header information, and a .raw file containing the raw bytes that make up the 3D array. Specific CT scans are identified using the series instance UID (series_uid) assigned when the CT scan was created. The coordinates are then to be transformed from the millimeter- based coordinate system (X, Y, Z) they’re expressed in, to the voxel-address-based coordinate system (I, R, C) used to take array slices from the CT scan data.

### <u> combining_data.py </u>
This includes functions and classes to handle data parsing, augmentation, and loading for training deep learning models. Below is a detailed overview of the main components.

### Libraries and Imports
- **Standard Libraries:** `copy`, `csv`, `functools`, `glob`, `os`, `random`, `math`, `collections.namedtuple`
- **Third-Party Libraries:** `SimpleITK`, `numpy`, `torch`
- **Custom Utilities:** `XyzTuple`, `xyz2irc` from `utill.util`, and `getCache` from `utill.disk`

### Global Variables and Constants
- **Logging Configuration:** Configured to log debug information.
- **Cache Initialization:** `raw_cache` initialized to cache raw CT scan data.

### Data Structures
- **`CandidateTuple`:** A named tuple storing details about potential nodule candidates:
  - `isNodule`: Boolean flag indicating if the candidate is a nodule.
  - `diameter_mm`: Diameter of the nodule in millimeters.
  - `series_uid`: Identifier for the CT scan series.
  - `center_xyz`: Tuple of coordinates indicating the center of the nodule.

### Functions and Classes

#### 1. `getCandidatesList` : Parses two CSV files (`annotations.csv` and `candidates.csv`) to generate a list of candidate nodules.
- **Input:** `reqOnDisk` (bool) - If `True`, only considers series that are available on disk.
- **Output:** A sorted list of `CandidateTuple` objects.

#### 2. `Ct` Class : Represents a CT scan and provides methods to retrieve specific chunks of the scan data.
- **Input:** `series_uid` (str) - Unique identifier for the CT scan series.
- **Attributes:** `series_uid`, `hunits_arr`, `origin_xyz`, `voxSize_xyz`, `direction_arr`
- **Methods:**
  - **`getCtChunk`:** 
    - **Input:** `center_xyz` (tuple), `width_irc` (tuple)
    - **Output:** `ct_chunk` (numpy array), `center_irc` (tuple)

#### 3. `getCt` : A cached function that returns a `Ct` object for the provided `series_uid`.
- **Input:** `series_uid` (str)
- **Output:** Returns a `Ct` instance.

#### 4. `getCtRawCandidate` : A cached function that retrieves a raw candidate chunk from the CT scan.
- **Input:** `series_uid` (str), `center_xyz` (tuple), `width_irc` (tuple)
- **Output:** `ct_chunk` (numpy array), `center_irc` (tuple)

#### 5. `getAugmenCandidate` : Generates an augmented candidate by applying transformations like mirroring, shifting, scaling, rotating, and adding noise.
- **Input:** `augmen_dict` (dict), `series_uid` (str), `center_xyz` (tuple), `width_irc` (tuple), `use_cache` (bool)
- **Output:** `augmented_chunk` (torch tensor), `center_irc` (tuple)

#### 6. `LunaDataset` Class : A PyTorch `Dataset` class that manages CT scan samples, providing access to both raw and augmented candidates for training and validation.
- **Input:** 
  - `val_step` (int) - Step for partitioning data into training and validation sets.
  - `isValset` (bool) - If `True`, the dataset is for validation.
  - `series_uid` (str) - Filter for a specific CT scan series.
  - `sortby_str` (str) - Sorting criteria for the candidates (`random`, `series_uid`, `label_and_size`).
  - `ratio_int` (int) - Ratio of negative to positive samples.
  - `augmen_dict` (dict) - Specifies augmentations.
  - `candidates_list` (list) - Predefined list of candidate tuples.
- **Methods:**
  - **`shuffleSamples`:** Shuffles the samples based on the positive to negative sample ratio.
  - **`__len__`:** Returns the number of samples.
  - **`__getitem__`:** Retrieves the candidate sample, applies augmentations, and returns as a tensor.

- **Output:** 
  - `candidate_tensor` (torch tensor) - Tensor of the candidate CT chunk.
  - `isNodule_tensor` (torch tensor) - Tensor indicating if the candidate is a nodule.
  - `series_uid` (str) - Identifier for the CT scan series.
  - `center_irc` (torch tensor) - Center of the candidate in array coordinates.
-------------------------

## Classification

Overview : This module is designed for the training of a 3D Convolutional Neural Network (CNN) model, specifically the `LunaModel`, which is implemented for binary classification tasks. The module handles the initialization of the model, optimizer, data loaders, and training loops. It also integrates TensorBoard for visualization and monitoring during training and validation.

---

### LunaModel Class
`LunaModel` is a 3D CNN model built with four convolutional blocks (`LunaBlock`), followed by a linear layer and a softmax activation function for classification.

#### Input:
- `inp_channels` (int): Number of input channels for the first convolutional layer. Default is `1`.
- `conv_channels` (int): Number of output channels for the first convolutional layer. This value doubles after each block. Default is `8`.

#### Output:
- `linear_output` (tensor): The raw output (logits) from the final linear layer.
- `act_output` (tensor): The softmax activated output, representing class probabilities.

#### Methods:
- `forward(input_batch)`: Forward pass through the network.
  - **Input**: `input_batch` (tensor) - The input tensor of shape `(batch_size, channels, depth, height, width)`.
  - **Output**: Returns a tuple containing `linear_output` and `act_output`.
- `_init_weights()`: Initializes the weights of the model using He initialization (Kaiming normal) for layers like `Conv3d` and `Linear`.

---

### LunaBlock Class

`LunaBlock` is a basic building block of the `LunaModel`, consisting of two 3D convolutional layers, each followed by a ReLU activation function, and a max-pooling layer.

---

### LunaTrainingApp Class

`LunaTrainingApp` handles the overall training process, including data loading, model initialization, training, validation, and logging.

#### Input:
- `sys_argv` (list of str, optional): Command-line arguments passed to the script. If `None`, uses `sys.argv`.

#### Methods:
- `__init__(sys_argv=None)`: Initializes the training application.
- `initModel()`: Initializes and returns the `LunaModel`.
  - **Output**: Returns the initialized model.
- `initOptimizer()`: Initializes the optimizer (SGD by default).
  - **Output**: Returns the optimizer.
- `initTrainDL()`: Initializes the training data loader.
  - **Output**: Returns the `DataLoader` for training data.
- `initValDL()`: Initializes the validation data loader.
  - **Output**: Returns the `DataLoader` for validation data.
- `initTensorboardWriters()`: Initializes TensorBoard writers for training and validation.
- `main()`: Main method for running the training and validation loops.
- `doTraining(epoch_index, train_dl)`: Executes the training loop for one epoch.
  - **Input**: `epoch_index` (int), `train_dl` (DataLoader) - The current epoch index and the training data loader.
  - **Output**: Returns the training metrics tensor.
- `doValidation(epoch_index, val_dl)`: Executes the validation loop for one epoch.
  - **Input**: `epoch_index` (int), `val_dl` (DataLoader) - The current epoch index and the validation data loader.
  - **Output**: Returns the validation metrics tensor.
- `computeBatchLoss(batch_index, batch_tuple, batch_size, metrics_gpu)`: Computes the loss for a batch during training or validation.
  - **Input**: `batch_index` (int), `batch_tuple` (tuple), `batch_size` (int), `metrics_gpu` (tensor).
  - **Output**: Returns the mean loss for the batch.
- `logMetrics(epoch_index, mode_str, metrics_tensor, classThreshold=0.5)`: Logs training or validation metrics to TensorBoard.
  - **Input**: `epoch_index` (int), `mode_str` (str), `metrics_tensor` (tensor), `classThreshold` (float).
  
---

### Command-Line Arguments

#### Arguments:
- `--num-workers` (int): Number of worker processes for data loading. Default is `8`.
- `--batch-size` (int): Batch size for training and validation. Default is `32`.
- `--epochs` (int): Number of epochs to train for. Default is `1`.
- `--balanced` (bool): If set, balances the training data. Default is `False`.
- `--augmented` (bool): Enables data augmentation. Default is `False`.
- `--augment-flip` (bool): Augments data by randomly flipping. Default is `False`.
- `--augment-offset` (bool): Augments data by offsetting along X and Y axes. Default is `False`.
- `--augment-scale` (bool): Augments data by scaling. Default is `False`.
- `--augment-rotate` (bool): Augments data by rotating around the head-foot axis. Default is `False`.
- `--augment-noise` (bool): Augments data by adding noise. Default is `False`.
- `--tb-prefix` (str): Prefix for TensorBoard logs. Default is `classification`.
- `comment` (str): Suffix for TensorBoard logs. Default is `lcd-pt`.

---

### TensorBoard Integration

The module integrates TensorBoard for tracking training progress, including metrics like loss, accuracy, precision, recall, and F1 score. It also logs histograms of predictions and precision-recall curves.

---

### Example Usage

To train the model using the script, run the following command:

```bash
python training.py --batch-size 64 --epochs 10 --augmented --augment-flip
```


--------------------------

## Segmentation
This file implements a structured approach to perform pixel-level segmentation of lung CT scans. It combines a UNet-based model with data augmentation and mask processing techniques to improve the accuracy of lung region segmentation. By handling the augmentation on the GPU and refining masks through various operations, the module ensures efficient and precise lung segmentation. The results are crucial for downstream tasks like detecting lung nodules or analyzing lung health.

## Why This Approach?

- **Lung Segmentation** is vital in medical imaging to focus on lung regions for analysis, especially in tasks like nodule detection. Accurate segmentation ensures that models concentrate on relevant areas of the scan.
- **Data Augmentation** improves model generalization by introducing variations in the input data, preventing overfitting.
- **Mask Processing** refines the model’s predictions to produce clean and accurate segmentation masks, crucial in medical imaging where precision is key.

---

## Class Breakdown

### `UNetWrapper`

This class encapsulates a UNet architecture and adds preprocessing (BatchNorm) and postprocessing (Sigmoid activation) for binary lung mask segmentation.

- **Purpose**: To segment lung regions from CT scan slices using a UNet model.
  
#### `__init__`
- **Input**:  
  - `kwargs`: UNet configuration, including the number of input channels (`in_channels`).
- **Output**:  
  - Initializes the UNet model with batch normalization and custom weight initialization.

#### `forward`
- **Input**:  
  - `input_batch`: A batch of CT scan images (shape: `[batch_size, channels, height, width]`).
- **Output**:  
  - Sigmoid-activated segmentation mask indicating the probability of each pixel being part of the lung.

---

### `SegmentationAugmentation`

This class handles various augmentations like flipping, scaling, rotation, and adding noise. The augmentations are applied to both the input images and their corresponding label masks to make the model robust.

- **Purpose**: To augment training data, improving the model’s ability to generalize to unseen data.

#### `__init__`
- **Input**:  
  - Augmentation parameters (`flip`, `offset`, `scale`, `rotate`, `noise`).

#### `forward`
- **Input**:  
  - `input_g`: Input image tensor.
  - `label_g`: Corresponding label tensor (lung masks).
- **Output**:  
  - `augmented_input_g`: Augmented images.
  - `augmented_label_g`: Augmented masks, transformed consistently with the images.

---

### `SegmentationMask`

This class refines segmentation masks using operations such as erosion, dilation, and cavity filling. These operations ensure that the lung masks are smooth, complete, and accurate, addressing common issues in raw mask outputs.

- **Purpose**: To refine lung segmentation masks using morphological operations for higher accuracy.

#### `__init__`
- **Output**:  
  - Initializes circular convolution layers for use in mask operations like erosion and dilation.

#### `erode` / `deposit`
- **Input**:  
  - `input_mask`: Input mask tensor.
  - `radius`: Radius of the circular kernel used for erosion or dilation.
  - `threshold`: Threshold for applying erosion or dilation.
- **Output**:  
  - Eroded or dilated mask.

#### `fill_cavity`
- **Input**:  
  - `input_mask`: Input mask tensor.
- **Output**:  
  - Mask with filled cavities, ensuring that all regions within the lung are segmented.

#### `forward`
- **Input**:  
  - `input_g`: Input image tensor.
  - `raw_pos_g`: Raw mask tensor.
- **Output**:  
  - `label_g`: Refined positive mask.
  - `neg_g`: Negative mask (non-lung regions).
  - `pos_g`: Positive mask (lung regions).
  - `lung_mask`: Final lung mask.
  - `mask_dict`: Dictionary containing all intermediate masks (`dense_mask`, `air_mask`, `candidate_mask`, etc.).

---

## Summary

This module brings together a UNet-based architecture, data augmentation, and mask processing to create an efficient and accurate lung segmentation system. It allows the model to generalize across different datasets while ensuring precision in segmentation, which is crucial for medical imaging tasks like lung nodule detection or disease diagnosis.




