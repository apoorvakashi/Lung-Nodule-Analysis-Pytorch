# LUNG NODULE ANALYSIS - PYTORCH 
<img width="753" alt="Screen Shot 2024-09-05 at 5 39 16 PM" src="https://github.com/user-attachments/assets/4bc2d536-c997-4450-b8f8-47596d335f65">


# Modules

## 1. Combining Data

The raw CT scan data has to be loaded into a form that can be used with PyTorch. The CT data comes in two files: a .mhd file containing metadata header information, and a .raw file containing the raw bytes that make up the 3D array. Specific CT scans are identified using the series instance UID (series_uid) assigned when the CT scan was created. The coordinates are then to be transformed from the millimeter- based coordinate system (X, Y, Z) they’re expressed in, to the voxel-address-based coordinate system (I, R, C) used to take array slices from the CT scan data.

### combining_data.py
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

#### 1. `getCandidatesList`
Parses two CSV files (`annotations.csv` and `candidates.csv`) to generate a list of candidate nodules.
- **Input:** `reqOnDisk` (bool) - If `True`, only considers series that are available on disk.
- **Output:** A sorted list of `CandidateTuple` objects.

#### 2. `Ct` Class
Represents a CT scan and provides methods to retrieve specific chunks of the scan data.
- **Input:** `series_uid` (str) - Unique identifier for the CT scan series.
- **Attributes:** `series_uid`, `hunits_arr`, `origin_xyz`, `voxSize_xyz`, `direction_arr`
- **Methods:**
  - **`getCtChunk`:** 
    - **Input:** `center_xyz` (tuple), `width_irc` (tuple)
    - **Output:** `ct_chunk` (numpy array), `center_irc` (tuple)

#### 3. `getCt`
A cached function that returns a `Ct` object for the provided `series_uid`.
- **Input:** `series_uid` (str)
- **Output:** Returns a `Ct` instance.

#### 4. `getCtRawCandidate`
A cached function that retrieves a raw candidate chunk from the CT scan.
- **Input:** `series_uid` (str), `center_xyz` (tuple), `width_irc` (tuple)
- **Output:** `ct_chunk` (numpy array), `center_irc` (tuple)

#### 5. `getAugmenCandidate`
Generates an augmented candidate by applying transformations like mirroring, shifting, scaling, rotating, and adding noise.
- **Input:** `augmen_dict` (dict), `series_uid` (str), `center_xyz` (tuple), `width_irc` (tuple), `use_cache` (bool)
- **Output:** `augmented_chunk` (torch tensor), `center_irc` (tuple)

### 6. `LunaDataset` Class
A PyTorch `Dataset` class that manages CT scan samples, providing access to both raw and augmented candidates for training and validation.
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


  


