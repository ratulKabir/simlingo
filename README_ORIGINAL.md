# [CVPR'25, Highlight] SimLingo: Vision-Only Closed-Loop Autonomous Driving with Language-Action Alignment

<p align="center">
  <h3 align="center">
    <a href="https://arxiv.org/abs/2503.09594"> Paper</a> | <a href="https://www.youtube.com/watch?v=Mpbnz2AKaNA&t=15s">Video</a> | <a href="https://www.katrinrenz.de/simlingo/">Website</a>
  </h3>
</p>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/carllava-vision-language-models-for-camera/carla-leaderboard-2-0-on-carla)](https://paperswithcode.com/sota/carla-leaderboard-2-0-on-carla?p=carllava-vision-language-models-for-camera)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/carllava-vision-language-models-for-camera/bench2drive-on-bench2drive)](https://paperswithcode.com/sota/bench2drive-on-bench2drive?p=carllava-vision-language-models-for-camera)

<p align="center" style="font-size:17px;">
SimLingo is a Vision-Language-Action (VLA) model that achieves state-of-the-art driving performance on the CARLA Leaderboard and Bench2Drive, while simultaniously including language capabilities like VQA, commentary, and instruction following.
</p>

<p align="center">
  <img src="assets/simlingo_teaser.png">
</p>



This repository is based on [Carla Garage](https://github.com/autonomousvision/carla_garage) and includes the PDM-lite expert, data collection code, language label generation, dreaming data generation, training of the base and final model, and evaluation of closed-loop driving and the language capabilities.

## 6-minute summary <a name="summary"></a> 

[<img src="assets/thumbnail.png" width="100%">](https://youtu.be/Mpbnz2AKaNA?si=qdQfhGIwnCbtD2DQ)


## News <a name="news"></a>
- **`[2025/05/08]`** Initial code release.
- **`[2025/04/28]`** SimLingo is accepted to CVPR as a highlight paper.


## Contents
1. [Setup](#setup)
2. [Dataset download (Coming Soon)](#dataset-download)
3. [Data Generation](#data-generation)
    - [Driving Data](#driving-data)
    - [Language Data](#language-data)
    - [Dreamer Data](#dreamer-data)
4. [Training](#training)
5. [Evaluation (Coming Soon)](#evaluation)
    - [Local Evaluation](#local-evaluation)
    - [Bench2Drive](#bench2drive)
    - [Language eval](#language-eval)
6. [Citations](#citations)
   
   
## Setup

Clone the repository, setup CARLA 0.9.15, and build the conda environment:
```Shell
git clone git@github.com:RenzKa/simlingo.git
cd simlingo
chmod +x setup_carla.sh
./setup_carla.sh
conda env create -f environment.yml
conda activate simlingo
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

## Dataset download
We will release our dataset with the Driving, VQA, Commentary and Dreamer labels soon.

## Dataset generation
### Driving Data

This repository uses the open source expert PDM-Lite from the paper [DriveLM](https://arxiv.org/abs/2312.14150) to generate the driving dataset. Most of the code for the data collection is taken from [Carla Garage](https://github.com/autonomousvision/carla_garage). However we changed some hyperparameter and used the data_agent from DriveLM which saves the required auxiliary information during data collection which is needed to generate the VQA and commentary data.

**Generate driving data:** To re-generate the data we provide a script for a SLURM cluster which parallelizes data collection across many GPUs (2080ti in our case). First, adjust the paths etc. in lines 213-230 of [collect_dataset_slurm.py](collect_dataset_slurm.py). You can specify the SLURM partition in [partition.txt](partition.txt) and change it during runtime. [max_num_jobs.txt](max_num_jobs.txt) specifies how many parallel SLURM jobs are submitted. This also can be changed during runtime. The data collection is started via `sbatch 0_run_collect_dataset_slurm.sh`, which calls `collect_dataset_slurm.py`. 
Increase the number in [max_num_jobs.txt](max_num_jobs.txt) once your setup works. 

**Dataset cleaning:** After the dataset is collected you can use `dataset_generation/delete_failed_runs.py` and `dataset_generation/delete_infraction_routes.py` to delete routes where the expert failed or carla crashed and the routes had to be restarted.

**Route files:** The routes for data collection are stored in [data/simlingo](data/simlingo/). **Note:** these are different route files as used in the Carla Garage. To generate our route files you can use the following script that generate our modified route files from the original carla route files: `bash dataset_generation/split_route_files.sh`
This splits the long training and validation route files provided by Carla into short routes with max 1 or 3 scenarios and balances and upsamples the scenarios.

PDM-Lite uses a modified version of the CARLA leaderboard that exposes additional information about the scenarios and makes data collection easier. They can be found in the [leaderboard_autopilot](leaderboard_autopilot) and [scenario_runner_autopilot](scenario_runner_autopilot) folders.

The dataset provided in this repository is not perfect. At some point while improving the model you will likely need to collect an improved version.

### Data buckets
Our bucketfile will be included in the released dataset. Coming soon.
If you want to generate your own buckets you can use the script `dataset_generation/data_buckets/carla_get_buckets.py`.

### Language Data
**VQA (DriveLM):** We use the script (with minor modifications) from [DriveLM](https://github.com/OpenDriveLab/DriveLM/tree/DriveLM-CARLA) to generate VQA labels. You can run `dataset_generation/language_labels/drivelm/carla_vqa_generator_main.py` to generate the VQA labels for your dataset. We used ChatGPT to augment the questions and answers. We provide the augmented templates, which we load during training in the folder [data/augmented_templates/drivelm_train_augmented_v2](data/augmented_templates/drivelm_train_augmented_v2). An example script to generate those augmented sentences can be found here: [dataset_generation/get_augmentations/gpt_augment_vqa.py](dataset_generation/get_augmentations/gpt_augment_vqa.py). **Note:** To be able to generate the VQA labels we save many auxiliary information of the simulator state during data collection. If you use a different dataset it is likely that this labelling script does not work.

**Commentary:** In this work we provide a new script to generate commentary labels. To generate commentary labels for your dataset, run `dataset_generation/language_labels/commentary/carla_commentary_generator_main.py`. We used ChatGPT to augment the questions and answers. We provide the augmented templates, which we load during training in the folder [data/augmented_templates/commentary_augmented.json](data/augmented_templates/commentary_augmented.json). Unfortunetly, based on how the project evolved the augmentations were first done manually for subsentences and later merged. If helpful we provide the subsentence level augmentationes [here](data/augmented_templates/commentary_subsentence.json) and the script to merge those to the final ones [here](dataset_generation/get_augmentations/commentary_merge_augmented.py). **Note:** To be able to generate the commentary labels we save many auxiliary information of the simulator state during data collection. If you use a different dataset it is likely that this labelling script does not work.

_File structure:_
``` bash
"image": # Path to rgb image
"commentary": # Commentary string (not augmented)
"commentary_template": # Commentary with placeholders for changing parts (e.g. object description, location). This is used to retrieve the augmentations.
"cause_object_visible_in_image": # Whether the object that causes the expert actions is visible in the front view image. Could be used to filter samples where the commentary describes an action based on a object not visible.
"cause_object": # Dictionary with attributes of the object causing the expert action.
"cause_object_string": # Language description of the cause object (e.g. dark green car that is to the front)
"scenario_name": # Name of the active CARLA scenario
"placeholder": # Dictionary to be able to replace the placeholders in commentary_template.
```

### Dreamer Data
To improve the alignment of language and actions we propose _Action Dreaming_ for which we provide a dataset with multiple different future trajectories given a language instruction. The language instructions cover a wide range of modes (e.g., speed changes, lane changes, object-centric navigation, crashes) with a label whether the execution is allowed and safe or not. 
To generate the labels run `dataset_generation/dreamer_data/dreamer_generator.py`.

_File structure:_
``` bash
category: # e.g. "target_speed", "stop", "faster", "crash", ...
      "waypoints": # Dreaming waypoints
      "route": # Dreaming path
      "rgb_path": # Path to rgb image in dataset
      "allowed": # Flag if execution is allowed.
      "mode": # category
      "info": # more information, e.g. about current, target and final speed
      "route_reasoning": # Language description about the route.
      "dreamer_instruction": # Language instruction.
      "instructions_templates": # Instruction with placeholders for changing parts (e.g. object description, location). This is used to retrieve the augmentations.
      "templates_placeholders": # Dictionary to be able to replace the placeholders in commentary_template.
      "dreamer_answer_safety": # Answer when safety mode is activated.
      "safe_to_execute": # Flag if the instruction is safe to execute.
```

## Training
We provide code for the smaller model SimLingo-Base (without language capabilities) in the folder `simlingo_base_training` and for the full model SimLingo in `simlingo_training`. For the config managment we use hydra. The config parameters are defined in the `config.py` file and can be adjusted in the `.yaml` files inside the `config` folder. 

We provide a SLURM script to start training: [train_simlingo_seed1.sh](train_simlingo_seed1.sh). This can be easily converted to a bash script to locally start the training. The entry file for training is [simlingo_training/train.py](simlingo_training/train.py).

With the default config the training logs in Wandb. Login is required. We also include a visualization callback that plots ground truth and predicted waypoints during training.


## Evaluation

Agent files for closed loop eval are coming soon.

### Bench2Drive
Bench2Drive is a CARLA benchmark proposed by the paper [Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-To-End Autonomous Driving](https://arxiv.org/abs/2406.03877). It consists of 220 very short (~150m) routes split across all towns with 1 safety critical scenario in each route.
Since it uses all towns for training, the methods have seen the test towns during training, so it can be considered a 'training' benchmark (reminiscent of level 4 driving).
The benchmark also comes with a training dataset generated by the [Think2Drive](https://arxiv.org/abs/2402.16720) expert, but we use the open-source expert [PDM-Lite](https://arxiv.org/abs/2312.14150) that achieves better resuslts and can be adapted to collect the necessary labels to produce VQA, Commentary and Dreamer data.
The benchmark and additional instructions can be found in the [Bench2Drive](Bench2Drive) folder.

**Start eval:** Evakuation on a SLURM cluster can be run with [start_eval_simlingo.py](start_eval_simlingo.py). The config dictionary needs to be adjusted with the correct names and paths. 

**Get results:** The script [Bench2Drive/tools/merge_route_json.py](Bench2Drive/tools/merge_route_json.py) can be used to obtain the final metrics after the evaluation is done. Make sure that all 220 routes are evaluated.

The Bench2Drive folder is based on version 0.0.3 of the [Bench2Drive repository](https://github.com/Thinklab-SJTU/Bench2Drive). Please cite the [Bench2Drive paper](https://arxiv.org/abs/2406.03877) when using the benchmark.

### Language eval
Coming soon.

## Citations
If you find this repository useful, please consider giving us a star &#127775;.
Please cite the following papers for the respective components of the repo:

SimLingo:
```BibTeX
@InProceedings{Renz2025cvpr,
  title={SimLingo: Vision-Only Closed-Loop Autonomous Driving with Language-Action Alignment},
  author={Renz, Katrin and Chen, Long and Arani, Elahe and Sinavski, Oleg},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

PDM-Lite expert:
```BibTeX
@inproceedings{Sima2024ECCV,
  title={DriveLM: Driving with Graph Visual Question Answering},
  author={Chonghao Sima and Katrin Renz and Kashyap Chitta and Li Chen and Hanxue Zhang and Chengen Xie and Jens Beißwenger and Ping Luo and Andreas Geiger and Hongyang Li},
  booktitle={Proc. of the European Conf. on Computer Vision (ECCV)},
  year={2024}
}
```

Bench2Drive benchmark:

```BibTeX
@inproceedings{Jia2024NeurIPS,
  title={Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-To-End Autonomous Driving},
  author={Xiaosong Jia and Zhenjie Yang and Qifeng Li and Zhiyuan Zhang and Junchi Yan},
  booktitle={NeurIPS 2024 Datasets and Benchmarks Track},
  year={2024}
}
```

## Other Resources
- [tuPlan garage](https://github.com/autonomousvision/tuplan_garage) | [CARLA garage](https://github.com/autonomousvision/carla_garage) | [Survey on E2EAD](https://github.com/OpenDriveLab/End-to-end-Autonomous-Driving)
- [DriveLM](https://github.com/OpenDriveLab/DriveLM/tree/main) | [PlanT](https://github.com/autonomousvision/plant) | [KING](https://github.com/autonomousvision/king) | [TransFuser](https://github.com/autonomousvision/transfuser) | [NEAT](https://github.com/autonomousvision/neat)

