from collections import Counter
from typing import List
import re 


class Tokenizer:
    def __init__(self , max_vocab_size: int = 20000 , padding_token:str = '<PAD>', unknown_token:str='<UNK>' , max_length:int = 1800):
        self.unknown_token = unknown_token
        self.padding_token = padding_token
        self.max_vocav_size = max_vocab_size
        self.max_length  = max_length
        self.word_to_idx = {}
        self.idx_to_word = {}

    def build_vocab(self, texts:List[str]):
        # Build the Vocabulary based on a list of text
        counter = Counter(word for text in texts for word in self._basic_tokenize(text))
        most_common_words = counter.most_common(self.max_vocav_size - 2 ) # Exclude the unknown and padd tokens 
        self.word_to_idx = {self.padding_token:self.max_vocav_size +1  , self.unknown_token:self.max_vocav_size+2 }
        self.word_to_idx.update({word:idx for idx , (word , _)  in enumerate(most_common_words)})
        self.idx_to_word = {idx:word for idx, word in self.word_to_idx.items()}

    def pad_sequence(self, sequence: List[int] ) -> List[int]:
         if len(sequence) < self.max_length:
            return sequence + [self.word_to_idx[self.padding_token]] * (self.max_length - len(sequence))
         else:
            return sequence[:self.max_length]
    def tokenize(self, text:str)-> List[str]:
        return self._basic_tokenize(text)
    
    def numericalize(self, tokens: List[str]) -> List[int]:
         return self.pad_sequence([self.word_to_idx.get(token ,    self.word_to_idx[self.unknown_token]) for token in tokens])
    
    def vocabulary(self):
        return self.word_to_idx

    def _basic_tokenize(self , text):
        
        text = text.lower()
        return re.findall(r"\b\w+\b", text)
