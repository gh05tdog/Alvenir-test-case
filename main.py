# Used for getting arguments from the command line
import argparse

# Used by the dataset builder and transcriber
import soundfile as sf
import torch
from datasets import load_dataset, Audio
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Tokenizer, Wav2Vec2Processor, \
    Wav2Vec2ForCTC

# Used to pick a random file from the dataset
import random

# Used to calculate the WER with the Levenshtein distance
from jiwer import wer

# Initialize parser
parser = argparse.ArgumentParser(
    # To define the name of the defult values in --help
formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# Adding optional argument
parser.add_argument("--model_id", default="Alvenir/wav2vec2-base-da-ft-nst", help="Model ID")
parser.add_argument("--data_path", default="Alvenir/alvenir_asr_da_eval", help="Path to data")

# Read arguments from command line
args = parser.parse_args()


class handler:
    def __init__(self, model_id, data_path):
        self.model_id = model_id
        self.data_path = data_path

    def run(self):
        print(self.model_id)
        print(self.data_path)

    def Load_data(self):
        print("Loading data")
        dataset = load_dataset(self.data_path, split="test").cast_column("audio", Audio(decode=False))
        return dataset[random.randint(0, len(dataset))]

    def transcribe(self):
        data = self.Load_data()
        print("Transcribing...")

        def load_model(model_path: str) -> Wav2Vec2ForCTC:
            return Wav2Vec2ForCTC.from_pretrained(model_path)

        def get_tokenizer(model_path: str) -> Wav2Vec2CTCTokenizer:
            return Wav2Vec2CTCTokenizer.from_pretrained(model_path)

        def get_processor(model_path: str) -> Wav2Vec2Processor:
            return Wav2Vec2Processor.from_pretrained(model_path)

        # Load model
        model = load_model(self.model_id)
        model.eval()
        tokenizer = get_tokenizer(self.model_id)
        processor = get_processor(self.model_id)
        audio, _ = sf.read(data["audio"]["path"])
        input_values = processor(audio, return_tensors="pt", padding="longest", sampling_rate=16_000).input_values
        with torch.no_grad():
            logistics = model(input_values).logits

        predicted_ids = torch.argmax(logistics, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        # Return transcription
        return transcription, data["sentence"]

    def compute_wer(self):
        com, sen = handler(args.model_id, args.data_path).transcribe()
        print("Computing WER")
        wer_error = wer(com, sen)
        return com, sen, wer_error


if __name__ == "__main__":
    computer, sentence, error = handler(args.model_id, args.data_path).compute_wer()

    print("The computer said: \" " + str(computer[0]) + " \" and the sentence was: \" " + str(sentence) + " \"" + "\n" \
          + "The WER is: " + str(error))
