import pandas as pd
import sentencepiece as spm

class AmharicTokenizer:
    def __init__(self, csv_file, text_column, vocab_size=1000, model_prefix='amharic_model'):
      
        try:
            self.csv_file = csv_file
            self.text_column = text_column
            self.vocab_size = vocab_size
            self.model_prefix = model_prefix

            # Write Amharic text to a temporary text file
            self.amharic_text_file = '../data/amharic_text.txt'
            self.write_amharic_text_to_file()

            # Train SentencePiece model
            self.train_sentencepiece_model()

            # Load trained SentencePiece model
            self.sp = spm.SentencePieceProcessor(model_file=f"{self.model_prefix}.model")

        except FileNotFoundError as fnfe:
            raise FileNotFoundError(f"CSV file '{csv_file}' not found.") from fnfe
        except Exception as e:
            raise ValueError(f"An error occurred during initialization: {e}")

    def write_amharic_text_to_file(self):
        df = pd.read_csv(self.csv_file)
        try:
            #  self.df = pd.read_csv(self.csv_file)
            amharic_text = df[self.text_column].astype(str).tolist()
            with open(self.amharic_text_file, 'w', encoding='utf-8') as file:
                file.writelines('\n'.join(amharic_text))
        except Exception as e:
            raise ValueError(f"An error occurred while writing Amharic text to file: {e}")

    def tokenize(self, amharic_text):
       
        try:
            return self.sp.encode_as_pieces(amharic_text)
        except Exception as e:
            raise ValueError(f"An error occurred during tokenization: {e}")

