# Rational LAMOL

Lifelong learning (LL) aims to train a neural network on a stream of tasks while retaining knowledge from previous tasks. However, many prior attempts in NLP still suffer from the catastrophic forgetting issue, where the model completely forgets what it just learned in the previous tasks.
In this paper, we introduce Rational LAMOL, a novel end-to-end LL framework for language models. In order to alleviate catastrophic forgetting, Rational LAMOL enhances LAMOL, a recent LL model, by applying critical freezing guided by human rationales. When the human rationales are not available, we propose exploiting unsupervised generated rationales as substitutions.  
 
** code mostly taken from [LAMOL](https://github.com/jojotenya/LAMOL) **

## Dataset
The datasets used in the experiment are Bool-Q, Movie Reviews, and SciFact.

| Dataset       | Download Link |
| ------------- |:-------------:|
| Bool-Q        | [ERASER](https://www.eraserbenchmark.com/) |
| Movie Reviews | [ERASER](https://www.eraserbenchmark.com/)      |
| SciFact       | [Link](https://drive.google.com/file/d/1j98m-7hlXfpemLMXukb9Kwnv023cGHai/view?usp=sharing)      |

## Training

Model training directly follows that of [LAMOL's](https://github.com/jojotenya/LAMOL) with a few distinctions.

#### Block level
To freeze critical block, run ``` train_freeze_block.py ``` with an additional argument ```--layer_to_freeze $LAYER``` where ```$LAYER``` is a transformer block index between 0-11.

#### Head level
To freeze critical heads, modify [this line](https://github.com/kanwatchara-k/r_lamol/blob/main/train_freeze_head.py#L233). The format of critical heads to be subjected to freezing is ```(layer_idx,[head_idx])``` e.g. ```(1,[1,2,3])``` means heads indices 1,2,3 of layer index 1 will be kept frozen.

## Critical Component Identification (CCI)
To identify critical component, run ```run_critical_freezing.py```

Currently, we've only experiment with using previous task's rationales to identify the component.

Arguments for CCI :
| Arguments       | Description |
| ------------- |:-------------:|
| head_level        | Do head level Granularity? |
| head_level_top_k  | Number of Heads to choose from      |
| data_dir          | Choice includes: movies,boolq,scifact. Data will be loaded from ```./data/{data_dir}/val.jsonl```      |
| old_model_dir     | The folder of the old model e.g. ./bms_model/boolq/|
| new_model_dir     | The folder of the new model e.g. ./bms_model/movies/|
| mo_gt_method      | Method to select from Model Old to Ground Truth |
| mn_mo_method      | Method to select from Model New to Model Old |
| device            | Device to use. CPU/GPU |
| n/n_ann           | Number of maximum annotations to do ie. 200 (We found that 200 is enough)|
| gen_rat        | Use generated rationale? |

## Unsupervised Rationale Generation
Any rationale generation module can be used. However, in this work we used [InvRat](https://github.com/code-terminator/invariant_rationalization).

Generated rationales has to be in the same format as ERASER's jsonl file and in the same directory as human rationales. Then simply run CCI with ```--gen_rat```.


TODO: \
~~Write Proper Readme~~ \
~~Upload Code~~ \
~~Upload SciFact dataset used in the paper~~
Refactor code, use submodule to properly give credit to LAMOL
