## About

This repository contains an a code of project "Vocoders for Text–to–Speech With GANs".

### Abstract
This paper explores strategies for accelerating GAN-based~\cite{gan} vocoders in text-to-speech systems, with a focus on optimizing inference speed without sacrificing audio quality. Building on established baselines such as HiFiGAN~\cite{hifigan} and LightVoc~\cite{lightvoc}, our work investigates architectural modifications and optimization techniques aimed specifically at reducing computation time. One possible approach to accelerating vocoders involves integrating state space models (SSMs~\cite{ssm}), particularly the Mamba framework~\cite{mamba}, to enhance processing efficiency. The study includes planned extensive experiments and evaluations—including both objective measurements and subjective listening tests—to assess the performance of our accelerated vocoders in terms of latency and audio fidelity compared to traditional GAN~\cite{gan} implementations. This work aims to achieve results that enable high-speed, real-time text-to-speech applications.




### Preavious HiFiGAN experiments
Best weights for model available: [here](https://disk.yandex.ru/d/DHq4RWzdd6x7DA)
Report on the completed work: [here](./report/report.md)

### Сurrent experiments

Wand Project: [here](https://wandb.ai/gsfatakhov-hse/TTS)

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
python3 resynthesis.py -cn=resynthesis HYDRA_CONFIG_ARGUMENTS
```

Where main `HYDRA_CONFIG_ARGUMENTS` are:

- `datasets.test.audio_path`: path to the ljspeech like dataset
- `inferencer.save_path`: path to the store results
- `inferencer.from_pretrained`: path to the weights of the model

#### To run synthesis, use the following command:

```bash
python3 synthesis.py -cn=synthesis HYDRA_CONFIG_ARGUMENTS
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

!Also you can use this command like this:

```bash
 python3 synthesize.py -cn=synthesize text="Hello world!"
```

instead of dataset path, you can use text to synthesize.
You get `single_utterance.wav` in the `results` folder.

## Other fearures

[Dataset](https://keithito.com/LJ-Speech-Dataset/)

## Credits

This repository is based on a heavily modified fork of [pytorch-template](https://github.com/victoresque/pytorch-template) and [asr_project_template](https://github.com/WrathOfGrapes/asr_project_template) repositories.

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
