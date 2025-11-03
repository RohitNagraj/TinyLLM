import regex as re
from typing import List, Dict, Tuple
from collections import defaultdict
from .helper import get_token_pair_count, merge_token_pairs, save_trained_tokenizer, load_trained_tokenizer

class TinyTokenizer:
    def __init__(self, train=False, model_path = "../models/minGPT.model") -> None:
        if train:
            self.merges = {}
            self.vocab = {}
            self.pattern =  r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        else:
            self.pattern, self.vocab, self.merges = load_trained_tokenizer(fp=model_path)
            
        self.compiled_pattern = re.compile(self.pattern)

    def train(self, text:str, vocab_size:int, verbose:bool = False, model_path = "../models/minGPT.model", vocab_path = "../models/minGPT.vocab"):
        """This method is used to train basic tokenizer wherein all the token ids are merged together

        Args:
            text (str): Training text data - wiki page of Kobe Bryant
            vocab_size (int): The size of the vocab size we want
            verbose (bool, optional): This is to keep the logging of data as print statements. Defaults to False.
        """
        text_chunks = re.findall(self.compiled_pattern, text)
        token_ids = [list(chunk.encode("utf-8")) for chunk in text_chunks]
        num_merges = vocab_size - 256
        vocab = {idx:bytes([idx]) for idx in range(256)}
        merges = {}
        for i in range(num_merges):
            overall_pairs = defaultdict(int)
            for chunk_ids in token_ids:
                get_token_pair_count(chunk_ids, overall_pairs)
                
            top_pair = max(overall_pairs, key=overall_pairs.get)
            idx = 256+i
            token_ids = [merge_token_pairs(tokens=chunk_ids, top_pair=top_pair, new_idx=idx) for chunk_ids in token_ids]
            if verbose:
                print(f"The following pairs were merged {top_pair} as {idx}")
            merges[top_pair] = idx
            vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]
        
        self.vocab = vocab
        self.merges = merges

        save_trained_tokenizer(self.pattern, self.vocab, self.merges, model_path, vocab_path)


    
    def _encode_chunk(self, text:str)->List[int]:
        """This function is used to encode any given text using the given vocabulary and the merged tokens

        Args:
            text (str): any piece of text that must be encoded

        Returns:
            int: List of token ids from the vocab
        """
        tokens = list(text.encode("utf-8"))
        while len(tokens)>=2:
            pairs = defaultdict(int)
            get_token_pair_count(tokens,pairs)
            min_pair = min(pairs, key=lambda x: self.merges.get(x, float("inf")))
            if min_pair not in self.merges:
                break
            idx = self.merges[min_pair]
            tokens = merge_token_pairs(tokens=tokens, top_pair=min_pair, new_idx=idx)
        
        return tokens
    
    def encode(self, text:str)->List[int]:
        """This function is used to encode the overall string and it calls the encode_chunk for encoding each chunk and must be joined to the overall ids.

        Args:
            text (str): any piece of text that must be chunked

        Returns:
            List[int]: List of token ids from the vocab
        """
        token_ids = []
        text_chunks = re.findall(self.compiled_pattern, text)
        for chunk in text_chunks:
            token_ids.extend(self._encode_chunk(chunk))
        
        return token_ids
        
    
    def decode(self, tokens:List[int])->str:
        """Given a list of token ids this function returns the string associated with the token ids

        Args:
            tokens (List[int]): List of ids given

        Returns:
            str: The output string of the text
        """
        tokens = b"".join(self.vocab[idx] for idx in tokens)
        text = tokens.decode("utf-8",errors="replace")
        return text