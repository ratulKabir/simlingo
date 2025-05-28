# SimLingo: Vision-Only Closed-Loop Autonomous Driving with Language-Action Alignment

This repository contains tools for collecting and processing autonomous driving data using the CARLA simulator on the local machine or AWS (no SLURM), including sensor data collection and language data generation. _Note that this repo is intended for the undertsanding of the workflow of SimLingo VLA, not to have a full on training pipeline. It can be too much to run this type of models on a single GPU machine with limited memory._ 

## Prerequisites


## Setup and Installation

### 1. CARLA Setup
Clone the repository, setup CARLA 0.9.15, and build the conda environment:
```Shell
git clone git@github.com:RenzKa/simlingo.git
cd simlingo
chmod +x setup_carla.sh
./setup_carla.sh
```

Before running the code, you will need to add the following paths to PYTHONPATH on your system:
```Shell
export CARLA_ROOT=/path/to/CARLA/root
export WORK_DIR=/path/to/simlingo
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}
```

### 2. Environment Setup

1. Create and activate a new conda environment:
   ```bash
   conda create -n simlingo python=3.8
   conda activate simlingo
   ```

2. Install PyTorch with CUDA support. Follow the [official link](https://pytorch.org/get-started/locally/).

3. (Have torch installed before running this step) Clone the repository:
   ```bash
   git clone [repository-url]
   cd simlingo
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Collection

Run the local data collection script:
```bash
python collect_dataset_local.py
```

The script will:
- Initialize CARLA simulation
- Process predefined routes and scenarios
- Collect sensor data, measurements, and route information
- Save data in the specified output directory

### Data Structure

Collected data will be saved in:
```
database/simlingo_v2_[date]/
├── raw_data/
│   ├── measurements/
│   ├── sensor_data/
│   └── route_info/
```

## Language Data Generation

After collecting the raw driving data, you can generate language labels:

### 1. VQA Generation

```bash
python dataset_generation/language_labels/generate_vqa.py
```

Generated VQA data will be saved in:
```
database/simlingo_v2_[date]/drivelmTD/
```

### 2. Commentary Generation

```bash
python dataset_generation/language_labels/generate_commentary.py
```

Generated commentary will be saved in:
```
database/simlingo_v2_[date]/commentary/
```

### 3. GPT Augmentation

To augment the language data using GPT-4:

```bash
python dataset_generation/get_augmentations/gpt_augment_vqa.py
```

### 4. Bucket Generation

To generate data buckets:
```bash
python dataset_generation/bucket_generation/generate_buckets.py
```

## License

This project is licensed under the [MIT License](LICENSE). 