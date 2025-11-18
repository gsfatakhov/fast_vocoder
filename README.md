## About

This repository contains the code for the project “Vocoders for Text-to-Speech with GANs”.

### Abstract
This paper explores strategies for accelerating GAN-based vocoders in text-to-speech systems, with a focus on optimizing inference speed without sacrificing audio quality. Building on established baselines such as HiFiGAN and LightVoc, our work investigates architectural modifications and optimization techniques aimed specifically at reducing computation time. One possible approach to accelerating vocoders involves integrating state space models (SSMs), particularly the Mamba framework, to enhance processing efficiency. The study includes planned extensive experiments and evaluations - including both objective measurements and subjective listening tests - to assess the performance of our accelerated vocoders in terms of latency and audio fidelity compared to traditional GAN implementations. This work aims to achieve results that enable high-speed, real-time text-to-speech applications.


### Metrics
<img width="1191" height="655" alt="image" src="https://github.com/user-attachments/assets/02b5a8e9-eea1-49ec-8179-2c4d596444ac" />

### MambaVocV1

Simple architecture. Experiments with hidden dimension, discriminators setup and [normalization](https://openreview.net/forum?id=YK8eO7BEkJ) (post/pred) respectively.
<img width="1191" height="655" alt="image" src="https://github.com/user-attachments/assets/497d4bce-6e16-45c9-9048-55277a377cf1" />
### MambaVocV2

Upsampling architecture. [Collaborative mechanism](https://arxiv.org/abs/2206.13404)
<img width="1191" height="655" alt="image" src="https://github.com/user-attachments/assets/1013c71c-0bc0-4ad1-a0b2-82b031d40f0e" />

### MambaVocV3
Combination of SSM, LN, Resblock. [Inspiration](inspiration)
<img width="1191" height="655" alt="image" src="https://github.com/user-attachments/assets/e550867d-be6f-4339-81e9-13715c034c19" />

### Baselines implementation
LightVoc did not converge to the desired metrics.
<img width="1191" height="655" alt="image" src="https://github.com/user-attachments/assets/5f8917e6-fd42-43b1-89a3-d4f681845a5d" />


### Experiments and weights links

Weights & Biases project: [here](https://wandb.ai/gsfatakhov-hse/TTS)


HiFiGAN weights:  [here](https://disk.yandex.ru/d/aqxgvqTKz0TRjA)


LightVoc weights: [here](https://disk.yandex.ru/d/New9Z6iowMRcgw)


MambaVocV3 weights: [here](https://disk.yandex.ru/d/nlR0FjgK5DTM5Q)

## Installation

The general steps are the following:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```


## How To Use

#### To train a model, run the following command:

```bash
python3 train.py -cn=train HYDRA_CONFIG_ARGUMENTS
```

Where `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

#### To run resynthesis, use the following command:

```bash
python3 resynthesize.py -cn=resynthesis HYDRA_CONFIG_ARGUMENTS
```

The main HYDRA_CONFIG_ARGUMENTS are:

- `datasets.test.audio_path`: path to the ljspeech like dataset
- `inferencer.save_path`: path where the results will be stored
- `inferencer.from_pretrained`: path to the weights of the model

#### To run synthesis, use the following command:

```bash
python3 synthesize.py -cn=synthesis HYDRA_CONFIG_ARGUMENTS
```

Where main `HYDRA_CONFIG_ARGUMENTS` are:

- `data.root_dir`: path to the dataset following format:

```bash
NameOfTheDirectoryWithUtterances
└── transcriptions
    ├── UtteranceID1.txt
    ├── UtteranceID2.txt
    .
    .
    .
    └── UtteranceIDn.txt
```

- `synthesizer.save_path`: path to the store results
- `synthesizer.from_pretrained`: path to the weights of the model

!Also you can use this command like this:code of project 

```bash
 python3 synthesize.py -cn=synthesize text="Hello world!"
```

instead of dataset path, you can use text to synthesize.
You get `single_utterance.wav` in the `results` folder.

## Other features

[Dataset](https://keithito.com/LJ-Speech-Dataset/)

### Previous HiFiGAN experiments
The best weights for the model are available: [here](https://disk.yandex.ru/d/DHq4RWzdd6x7DA)
Report on the completed work: [here](./report/report.md)

## Credits

This repository is based on a heavily modified fork of [pytorch-template](https://github.com/victoresque/pytorch-template) and [asr_project_template](https://github.com/WrathOfGrapes/asr_project_template) repositories.

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
