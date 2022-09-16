# Alvenir-test-case

This project was made as a supplement to a job interview. 
It is a python module for transcribing audio files and evaluating model performance.

The program picks a "random" sample from the DATA_PATH and transcibes it. Lastly it uses [levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance) to compute the word error rate. 

The program will display the transcribtion, the real sentence and the WER (Word Error Rate) in the console.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requrements.txt.

```bash
pip install -r requirements.txt
```

## Usage
Run main.py (with or without arguments)

```python
main.py
```
If you want to use the arguments you should use:

```bash
usage: main.py [-h] [--model_id MODEL_ID] [--data_path DATA_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --model_id MODEL_ID   Model ID (default: Alvenir/wav2vec2-base-da-ft-nst)
  --data_path DATA_PATH
                        Path to data (default: Alvenir/alvenir_asr_da_eval)

```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Open souce are of course the best, and i hope you have the same mindset :D
