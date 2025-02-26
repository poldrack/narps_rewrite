### Rewrite of NARPS analysis codebase


### Claude 3.7 (Thinking) analysis and suggested rewrite

> Attached is code from a scientific data analysis project.  Please explain the purpose of each of the classes in this file.  Then, explain the purpose of each module in each class. For each module, please outline the inputs to the module (both in terms of function arguments and any files loaded by the module) and outputs from the model (both in terms of returned values and any files written out)

<think>Let me analyze the code and break down the classes and their methods, focusing on inputs and outputs for each.

The file `narps.py` contains three main classes:
1. `NarpsDirs` - For managing directories
2. `NarpsTeam` - For managing data from individual teams
3. `Narps` - The main class for the NARPS (Neuroimaging Analysis Replication and Prediction Study) analysis

Let me break down each class and its methods:</think># Analysis of NARPS Python Code

The code you've shared is from a neuroimaging analysis project called NARPS (Neuroimaging Analysis Replication and Prediction Study). This appears to be a meta-analysis framework for comparing neuroimaging results across multiple research teams.

## Classes Overview

The code defines three main classes:

1. **NarpsDirs**: Manages the directory structure for the project
2. **NarpsTeam**: Represents an individual research team's data
3. **Narps**: The main class that manages the entire dataset and analysis workflow

Let's examine each in detail:

## 1. NarpsDirs Class

This class manages all directories needed for analysis.

### Methods:

#### `__init__(self, basedir=None, dataurl=None, force_download=False, testing=False)`
- **Purpose**: Initialize the directory structure
- **Inputs**:
  - `basedir`: Base directory for the project (can use NARPS_BASEDIR environment variable)
  - `dataurl`: URL to download data from (can use DATA_URL environment variable)
  - `force_download`: Whether to force redownload of data
  - `testing`: Flag for testing mode
- **Outputs**: 
  - Creates directory structure in filesystem
  - Sets up `self.dirs` dictionary with paths
  - Creates log file

#### `get_output_dir(self, dirID, base='output')`
- **Purpose**: Get or create a directory path
- **Inputs**: 
  - `dirID`: Directory identifier
  - `base`: Base directory name
- **Outputs**:
  - Returns directory path
  - Creates directory if it doesn't exist

#### `get_orig_data(self)`
- **Purpose**: Download original data from repository
- **Inputs**: Uses `self.data_url` from initialization
- **Outputs**:
  - Downloads and extracts data to `self.dirs['base']`
  - Logs hash of downloaded file

## 2. NarpsTeam Class

This class represents a single team's data and analysis.

### Methods:

#### `__init__(self, teamID, NV_collection_id, dirs, verbose=False)`
- **Purpose**: Initialize team data structure
- **Inputs**:
  - `teamID`: Team identifier
  - `NV_collection_id`: NeuroVault collection ID
  - `dirs`: NarpsDirs object with directory information
  - `verbose`: Flag for verbose output
- **Outputs**:
  - Sets up file paths and data structures

#### `get_orig_images(self)`
- **Purpose**: Locate original images for team
- **Inputs**: Uses team information from initialization
- **Outputs**: 
  - Populates `self.images` dictionary with file paths
  - Sets `self.has_all_images` flags

#### `create_binarized_thresh_masks(self, thresh=1e-4, overwrite=False, replace_na=True)`
- **Purpose**: Create binary masks from thresholded maps
- **Inputs**:
  - `thresh`: Threshold value
  - `overwrite`: Whether to overwrite existing files
  - `replace_na`: Whether to replace NaN values
- **Outputs**:
  - Creates binarized mask files in the threshold directory
  - Updates `self.images` with paths to new files

#### `get_resampled_images(self, imgtype, overwrite=False, replace_na=False)`
- **Purpose**: Resample images to common space
- **Inputs**:
  - `imgtype`: Image type ('thresh' or 'unthresh')
  - `overwrite`: Whether to overwrite existing files
  - `replace_na`: Whether to replace NaN values
- **Outputs**:
  - Creates resampled image files
  - Updates `self.images` with paths to resampled files

## 3. Narps Class

This is the main class orchestrating the entire analysis.

### Methods:

#### `__init__(self, basedir=None, metadata_file=None, verbose=False, overwrite=False, dataurl=None, testing=False)`
- **Purpose**: Initialize main analysis object
- **Inputs**:
  - `basedir`: Base directory (can use NARPS_BASEDIR environment variable)
  - `metadata_file`: Path to metadata file
  - `verbose`: Flag for verbose output
  - `overwrite`: Whether to overwrite existing files
  - `dataurl`: URL for data download
  - `testing`: Flag for testing mode
- **Outputs**:
  - Sets up NarpsDirs
  - Creates full mask image if needed
  - Gets input directories

#### `mk_full_mask_img(self, dirs)`
- **Purpose**: Create a mask with ones in all voxels
- **Inputs**: `dirs`: NarpsDirs object
- **Outputs**: Creates full mask image file

#### `get_input_dirs(self, dirs, verbose=True, load_json=True)`
- **Purpose**: Find input directories with team data
- **Inputs**:
  - `dirs`: NarpsDirs object
  - `verbose`: Flag for verbose output
  - `load_json`: Whether to load JSON metadata
- **Outputs**:
  - Populates `self.teams` dictionary with NarpsTeam objects
  - Logs number of input directories found

#### `get_orig_images(self, dirs)`
- **Purpose**: Load original images for all teams
- **Inputs**: `dirs`: NarpsDirs object
- **Outputs**:
  - Populates `self.complete_image_sets` with teams having complete data
  - Sorts teams for consistent order

#### `get_binarized_thresh_masks(self)`
- **Purpose**: Create binary masks for thresholded maps
- **Inputs**: Uses team information from initialization
- **Outputs**: Creates binarized masks for each team

#### `get_resampled_images(self, overwrite=None)`
- **Purpose**: Resample all images to FSL MNI space
- **Inputs**: `overwrite`: Whether to overwrite existing files
- **Outputs**: Creates resampled images for each team

#### `check_image_values(self, overwrite=None)`
- **Purpose**: Count nonzero and NA voxels in images
- **Inputs**: `overwrite`: Whether to overwrite existing metadata
- **Outputs**:
  - Creates or loads `image_metadata_df.csv` with voxel counts
  - Returns DataFrame with image metadata

#### `create_concat_images(self, datatype='resampled', create_voxel_map=False, imgtypes=None, overwrite=None)`
- **Purpose**: Concatenate images across teams
- **Inputs**:
  - `datatype`: Data type to use
  - `create_voxel_map`: Whether to create voxel proportion maps
  - `imgtypes`: Image types to process
  - `overwrite`: Whether to overwrite existing files
- **Outputs**:
  - Creates concatenated image files
  - Creates voxel maps if requested
  - Writes label files with team information

#### `create_mean_thresholded_images(self, datatype='resampled', overwrite=None, thresh=1e-5)`
- **Purpose**: Create overlap maps for thresholded images
- **Inputs**:
  - `datatype`: Data type to use
  - `overwrite`: Whether to overwrite existing files
  - `thresh`: Threshold value
- **Outputs**: Creates mean thresholded image files

#### `create_rectified_images(self, map_metadata_file=None, overwrite=None)`
- **Purpose**: Create rectified images (flip negative contrasts)
- **Inputs**:
  - `map_metadata_file`: Path to map metadata
  - `overwrite`: Whether to overwrite existing files
- **Outputs**:
  - Creates rectified image files
  - Writes list of rectified images

#### `compute_image_stats(self, datatype='zstat', overwrite=None)`
- **Purpose**: Compute statistics on images
- **Inputs**:
  - `datatype`: Data type to use
  - `overwrite`: Whether to overwrite existing files
- **Outputs**: Creates range and standard deviation image files

#### `convert_to_zscores(self, map_metadata_file=None, overwrite=None)`
- **Purpose**: Convert images to Z-scores
- **Inputs**:
  - `map_metadata_file`: Path to map metadata
  - `overwrite`: Whether to overwrite existing files
- **Outputs**: Creates Z-score image files

#### `estimate_smoothness(self, overwrite=None, imgtype='zstat')`
- **Purpose**: Estimate smoothness of Z maps using FSL
- **Inputs**:
  - `overwrite`: Whether to overwrite existing files
  - `imgtype`: Image type to process
- **Outputs**:
  - Creates `smoothness_est.csv` with smoothness estimates
  - Returns DataFrame with smoothness data

#### `write_data(self, save_data=True, outfile=None)`
- **Purpose**: Serialize important information
- **Inputs**:
  - `save_data`: Whether to save data
  - `outfile`: Output file path
- **Outputs**:
  - Creates pickle file with analysis state
  - Returns dictionary with information

#### `load_data(self, infile=None)`
- **Purpose**: Load serialized data
- **Inputs**: `infile`: Input file path
- **Outputs**: Restores analysis state from file

This code appears to implement a framework for analyzing neuroimaging results from multiple teams working on the same dataset, focusing on standardizing, transforming, and comparing their statistical maps.

> I would like to rewrite this code to make it more modular and easy to understand and test.  can you please recommend how I might do this?

<think>Let's analyze the existing code structure and recommend a more modular, maintainable, and testable approach. The current code has several issues:

1. Large classes with many responsibilities
2. Methods that are too long and do multiple things
3. Tight coupling between components
4. Limited separation of concerns
5. Hardcoded values and paths
6. Limited error handling and validation
7. Lack of type hints

Here's my approach to recommending improvements:</think># Recommendations for Refactoring the NARPS Code

After analyzing the codebase, I can recommend several strategies to make the code more modular, easier to understand, and more testable. The current code has some common issues that can be addressed through refactoring:

## 1. Apply the Single Responsibility Principle

The current classes are very large with many responsibilities. I recommend:

- **Break down the large classes**: Split the `Narps` class into multiple smaller classes with specific responsibilities.
- **Create dedicated modules**: For example, separate the image processing logic, file management, and analysis into their own modules.

```
narps/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── config.py          # Configuration handling
│   └── directories.py     # Directory management (refactored from NarpsDirs)
├── data/
│   ├── __init__.py
│   ├── downloader.py      # Data download functionality
│   ├── team.py            # Team data handling (refactored from NarpsTeam)
│   └── validation.py      # Data validation functions
├── imaging/
│   ├── __init__.py
│   ├── processing.py      # Image processing operations
│   ├── statistics.py      # Statistical operations on images
│   └── visualization.py   # Visualization functions
├── analysis/
│   ├── __init__.py
│   ├── rectification.py   # Contrast rectification functions
│   ├── smoothness.py      # Smoothness estimation
│   └── aggregation.py     # Functions for combining results across teams
└── utils/
    ├── __init__.py
    ├── logging.py         # Logging utilities
    └── serialization.py   # Save/load functionality
```

## 2. Use Dependency Injection

Rather than having classes create their dependencies internally:

```python
# Instead of this:
class Narps:
    def __init__(self, basedir=None, ...):
        self.dirs = NarpsDirs(basedir, dataurl=dataurl)
        
# Do this:
class Narps:
    def __init__(self, dirs, ...):
        self.dirs = dirs
```

This makes testing much easier as you can mock dependencies.

## 3. Separate I/O from Processing Logic

Create clear boundaries between code that:
1. Reads/writes files
2. Processes data
3. Manages program flow

For example:

```python
# Instead of:
def create_binarized_thresh_masks(self):
    # ... file loading, processing, and writing all mixed together
    
# Refactor into:
def load_thresh_image(path):
    """Load threshold image from path"""
    # loading only

def create_binary_mask(image_data, threshold, rectify=False):
    """Pure function to create binary mask from image data"""
    # processing only

def save_binary_mask(mask_data, path):
    """Save binary mask to file"""
    # saving only

def process_team_masks(team, threshold=1e-4):
    """Orchestration function"""
    for hyp in range(1, 10):
        img = load_thresh_image(team.images['thresh']['orig'][hyp])
        binary_mask = create_binary_mask(img, threshold, team.rectify[hyp])
        save_binary_mask(binary_mask, team.get_mask_path(hyp))
```

## 4. Add Type Hints and Documentation

Use type hints to make the code more self-documenting and amenable to static analysis:

```python
from typing import Dict, List, Optional, Union
import numpy as np
import nibabel as nib

def create_binary_mask(
    image_data: np.ndarray, 
    threshold: float, 
    rectify: bool = False
) -> np.ndarray:
    """
    Create a binary mask from image data.
    
    Args:
        image_data: Input image data
        threshold: Threshold value for binarization
        rectify: Whether to reverse contrast direction
        
    Returns:
        Binary mask as numpy array
    """
    # function implementation
```

## 5. Use Configuration Management

Replace hardcoded values with a configuration system:

```python
# config.py
import os
from dataclasses import dataclass

@dataclass
class NarpsConfig:
    base_dir: str = os.getenv('NARPS_BASEDIR', '')
    data_url: str = os.getenv('DATA_URL', '')
    fsl_dir: str = os.getenv('FSLDIR', '')
    threshold: float = 1e-4
    hypotheses: Dict[int, str] = field(default_factory=lambda: {
        1: '+gain: equal indiff',
        # ... other hypotheses
    })
```

## 6. Use Context Managers for Resource Handling

Ensure resources are properly closed, especially for file operations:

```python
def save_to_json(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f)
```

## 7. Implement Better Error Handling

Add proper error handling with specific exception types:

```python
class NarpsError(Exception):
    """Base exception for NARPS errors"""
    pass

class DataDownloadError(NarpsError):
    """Error during data download"""
    pass

def download_data(url, destination):
    try:
        # download code
    except HTTPError as e:
        raise DataDownloadError(f"Failed to download data: {e}") from e
```

## 8. Create a Command-Line Interface

For better usability, consider adding a command-line interface:

```python
# cli.py
import argparse
import sys
from narps.core.config import NarpsConfig
from narps.core.directories import DirectoryManager
from narps.analysis.pipeline import AnalysisPipeline

def main():
    parser = argparse.ArgumentParser(description="NARPS Analysis Tool")
    parser.add_argument("--basedir", help="Base directory for analysis")
    parser.add_argument("--metadata", help="Path to metadata file")
    # ... other arguments
    
    args = parser.parse_args()
    config = NarpsConfig()
    if args.basedir:
        config.base_dir = args.basedir
    
    # Setup
    dirs = DirectoryManager(config)
    pipeline = AnalysisPipeline(dirs, args.metadata)
    
    # Run analysis
    pipeline.run()

if __name__ == "__main__":
    main()
```

## 9. Implement Proper Testing

Refactor to make testing easier:

```python
# test_binary_mask.py
import numpy as np
import pytest
from narps.imaging.processing import create_binary_mask

def test_create_binary_mask_positive():
    # Create test data
    test_data = np.array([[-1, 0, 1], [2, 3, 4]])
    
    # Test with positive threshold
    result = create_binary_mask(test_data, threshold=0.5, rectify=False)
    expected = np.array([[0, 0, 1], [1, 1, 1]])
    np.testing.assert_array_equal(result, expected)

def test_create_binary_mask_rectified():
    # Create test data
    test_data = np.array([[-1, 0, 1], [2, 3, 4]])
    
    # Test with rectification
    result = create_binary_mask(test_data, threshold=0.5, rectify=True)
    expected = np.array([[1, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(result, expected)
```

## 10. Use Logging Instead of Print Statements

Replace print statements with proper logging:

```python
import logging

logger = logging.getLogger(__name__)

def process_team_data(team_id):
    logger.info(f"Processing team {team_id}")
    try:
        # processing code
        logger.debug("Data processing completed")
    except Exception as e:
        logger.error(f"Error processing team {team_id}: {e}")
        raise
```

## Implementation Strategy

I recommend a phased approach:

1. Create the new package structure and gradually migrate functionality.
2. Start with the most self-contained parts (utilities, file operations).
3. Write tests for the new code as you go.
4. Make the original code call your new modules until you can completely replace it.
5. Add documentation throughout the process.

This approach allows you to maintain a working system while improving it incrementally, rather than rewriting everything at once.



### Claude 3.5 Sonnet analysis and suggested rewrite.

> Attached is code from a scientific data analysis project.  Please explain the purpose of each of the classes in this file.  Then, explain the purpose of each module in each class. For each module, please outline the inputs to the module (both in terms of function arguments and any files loaded by the module) and outputs from the model (both in terms of returned values and any files written out)


I'll analyze each class and its modules in detail.

### Classes Overview

1. **NarpsDirs**: Manages directories for the NARPS (Neuroimaging Analysis Replication and Prediction Study) project
2. **NarpsTeam**: Handles data and operations for individual research teams
3. **Narps**: Main class that orchestrates the entire analysis workflow

### Detailed Class Analysis

#### 1. NarpsDirs Class
Purpose: Manages directory structure and data downloads for the project

Key Methods:

**__init__(basedir=None, dataurl=None, force_download=False, testing=False)**
- Inputs:
  - basedir: Base directory path
  - dataurl: URL for downloading data
  - force_download: Whether to force redownload
  - testing: Testing mode flag
- Outputs:
  - Creates directory structure
  - Sets up class attributes for directory paths

**get_output_dir(dirID, base='output')**
- Inputs:
  - dirID: Directory identifier
  - base: Base directory name
- Outputs:
  - Returns path to output directory
  - Creates directory if it doesn't exist

**get_orig_data()**
- Inputs:
  - Uses self.data_url
- Outputs:
  - Downloads and extracts original data
  - Creates log file with MD5 hash

#### 2. NarpsTeam Class
Purpose: Manages data and operations for individual research teams

Key Methods:

**__init__(teamID, NV_collection_id, dirs, verbose=False)**
- Inputs:
  - teamID: Team identifier
  - NV_collection_id: NeuroVault collection ID
  - dirs: NarpsDirs instance
  - verbose: Verbosity flag
- Outputs:
  - Initializes team-specific data structures

**get_orig_images()**
- Inputs:
  - Uses self.input_dir
- Outputs:
  - Populates self.images with paths to original images
  - Sets self.has_all_images flag

**create_binarized_thresh_masks(thresh=1e-4, overwrite=False, replace_na=True)**
- Inputs:
  - thresh: Threshold value
  - overwrite: Whether to overwrite existing files
  - replace_na: Whether to replace NaN values
- Outputs:
  - Creates binarized threshold masks
  - Saves to thresh_mask_orig directory

**get_resampled_images(imgtype, overwrite=False, replace_na=False)**
- Inputs:
  - imgtype: Type of image ('thresh' or 'unthresh')
  - overwrite: Whether to overwrite existing files
  - replace_na: Whether to replace NaN values
- Outputs:
  - Creates resampled images in common space
  - Saves to resampled directory

#### 3. Narps Class
Purpose: Main class orchestrating the entire analysis workflow

Key Methods:

**__init__(basedir=None, metadata_file=None, verbose=False, overwrite=False, dataurl=None, testing=False)**
- Inputs:
  - basedir: Base directory path
  - metadata_file: Path to metadata file
  - verbose: Verbosity flag
  - overwrite: Whether to overwrite existing files
  - dataurl: URL for data download
  - testing: Testing mode flag
- Outputs:
  - Initializes analysis environment

**mk_full_mask_img(dirs)**
- Inputs:
  - dirs: NarpsDirs instance
- Outputs:
  - Creates full brain mask image

**get_input_dirs(dirs, verbose=True, load_json=True)**
- Inputs:
  - dirs: NarpsDirs instance
  - verbose: Verbosity flag
  - load_json: Whether to load JSON metadata
- Outputs:
  - Identifies and catalogs input directories
  - Creates team instances

**create_concat_images(datatype='resampled', create_voxel_map=False, imgtypes=None, overwrite=None)**
- Inputs:
  - datatype: Type of data to concatenate
  - create_voxel_map: Whether to create voxel maps
  - imgtypes: Types of images to process
  - overwrite: Whether to overwrite existing files
- Outputs:
  - Creates concatenated images across teams
  - Optionally creates voxel maps

**compute_image_stats(datatype='zstat', overwrite=None)**
- Inputs:
  - datatype: Type of data to analyze
  - overwrite: Whether to overwrite existing files
- Outputs:
  - Computes standard deviation and range statistics
  - Saves results to output directories

**convert_to_zscores(map_metadata_file=None, overwrite=None)**
- Inputs:
  - map_metadata_file: Path to map metadata
  - overwrite: Whether to overwrite existing files
- Outputs:
  - Converts t-statistics to z-scores
  - Saves converted images

**estimate_smoothness(overwrite=None, imgtype='zstat')**
- Inputs:
  - overwrite: Whether to overwrite existing files
  - imgtype: Type of image to analyze
- Outputs:
  - Estimates image smoothness using FSL
  - Saves results to CSV file

**write_data(save_data=True, outfile=None)** and **load_data(infile=None)**
- Handle serialization and deserialization of analysis state
- Inputs/Outputs:
  - Pickle files containing analysis state and results

This codebase appears to be designed for processing and analyzing neuroimaging data from multiple research teams, with a focus on standardization, quality control, and statistical analysis of the imaging results.

> I would like to rewrite this code to make it more modular and easy to understand and test.  can you please recommend how I might do this?


I'll suggest several ways to improve the code's modularity, testability, and maintainability:

1. **Separate Concerns Using Multiple Modules**
First, I'd split the code into separate modules based on functionality:

```
narps/
├── __init__.py
├── config/
│   ├── __init__.py
│   └── settings.py           # Configuration settings
├── core/
│   ├── __init__.py
│   ├── directories.py        # NarpsDirs functionality
│   ├── team.py              # NarpsTeam functionality
│   └── analysis.py          # Main analysis functionality
├── utils/
│   ├── __init__.py
│   ├── io.py                # File I/O operations
│   ├── image.py             # Image processing utilities
│   └── validation.py        # Data validation utilities
└── tests/
    ├── __init__.py
    ├── test_directories.py
    ├── test_team.py
    └── test_analysis.py
```

2. **Use Dependency Injection**
Let's modify the class structure to make dependencies explicit and more testable:

```python
# core/directories.py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

@dataclass
class DirectoryConfig:
    base_dir: Path
    data_url: str
    force_download: bool = False
    testing: bool = False

class DirectoryManager:
    def __init__(self, config: DirectoryConfig):
        self.config = config
        self.dirs: Dict[str, Path] = {}
        self._initialize_directories()

    def _initialize_directories(self) -> None:
        # Directory initialization logic
        pass

    def get_output_dir(self, dir_id: str) -> Path:
        # Directory creation/retrieval logic
        pass
```

3. **Use Type Hints and Data Classes**
```python
# core/team.py
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

@dataclass
class TeamConfig:
    team_id: str
    collection_id: str
    input_dir: Path
    verbose: bool = False

class TeamData:
    def __init__(self, config: TeamConfig, dir_manager: DirectoryManager):
        self.config = config
        self.dir_manager = dir_manager
        self._initialize_data()

    def process_images(self) -> Dict[str, Path]:
        # Image processing logic
        pass
```

4. **Create a Pipeline Class for Analysis**
```python
# core/analysis.py
from typing import List, Dict
from pathlib import Path

class AnalysisPipeline:
    def __init__(self, teams: List[TeamData], dir_manager: DirectoryManager):
        self.teams = teams
        self.dir_manager = dir_manager

    def run_analysis(self) -> Dict:
        results = {}
        for step in self._get_analysis_steps():
            results.update(step())
        return results

    def _get_analysis_steps(self):
        return [
            self._preprocess_images,
            self._compute_statistics,
            self._generate_reports
        ]
```

5. **Add Configuration Management**
```python
# config/settings.py
from pathlib import Path
from typing import Dict
import yaml

class Settings:
    def __init__(self, config_path: Path):
        self.config = self._load_config(config_path)

    @staticmethod
    def _load_config(config_path: Path) -> Dict:
        with open(config_path) as f:
            return yaml.safe_load(f)
```

6. **Add Proper Error Handling and Logging**
```python
# utils/validation.py
from typing import Protocol
from pathlib import Path
import logging

class DataValidator(Protocol):
    def validate(self) -> bool:
        ...

class ImageValidator(DataValidator):
    def __init__(self, image_path: Path):
        self.image_path = image_path
        self.logger = logging.getLogger(__name__)

    def validate(self) -> bool:
        try:
            # Validation logic
            return True
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return False
```

7. **Create a Factory for Object Creation**
```python
# core/factory.py
from typing import Dict
from pathlib import Path

class NarpsFactory:
    @staticmethod
    def create_directory_manager(config: Dict) -> DirectoryManager:
        dir_config = DirectoryConfig(**config)
        return DirectoryManager(dir_config)

    @staticmethod
    def create_team(config: Dict, dir_manager: DirectoryManager) -> TeamData:
        team_config = TeamConfig(**config)
        return TeamData(team_config, dir_manager)
```

8. **Main Entry Point**
```python
# __main__.py
from narps.config.settings import Settings
from narps.core.factory import NarpsFactory
from narps.core.analysis import AnalysisPipeline

def main():
    # Load configuration
    settings = Settings(Path("config.yaml"))
    
    # Create directory manager
    dir_manager = NarpsFactory.create_directory_manager(settings.config["directories"])
    
    # Create teams
    teams = [
        NarpsFactory.create_team(team_config, dir_manager)
        for team_config in settings.config["teams"]
    ]
    
    # Create and run pipeline
    pipeline = AnalysisPipeline(teams, dir_manager)
    results = pipeline.run_analysis()

if __name__ == "__main__":
    main()
```

9. **Add Tests**
```python
# tests/test_team.py
import pytest
from narps.core.team import TeamData, TeamConfig
from narps.core.directories import DirectoryManager

def test_team_image_processing():
    # Setup
    config = TeamConfig(...)
    dir_manager = DirectoryManager(...)
    team = TeamData(config, dir_manager)
    
    # Test
    results = team.process_images()
    
    # Assert
    assert len(results) > 0
    assert all(path.exists() for path in results.values())
```

These changes would provide several benefits:

1. **Better Separation of Concerns**: Each module has a single responsibility
2. **Improved Testability**: Dependencies are explicit and can be mocked
3. **Type Safety**: Type hints help catch errors early
4. **Configuration Management**: Settings are centralized and validated
5. **Error Handling**: Consistent error handling and logging
6. **Dependency Injection**: Makes the code more flexible and testable
7. **Factory Pattern**: Centralizes object creation
8. **Pipeline Pattern**: Makes the analysis flow clear and maintainable

Would you like me to elaborate on any of these aspects or provide more specific examples for any particular part?
