from typing import List, Dict, Tuple
import unicodedata

def read_train_data(filepath:str)->str:
    """This function is used to read the training data

    Args:
        filepath (str): The location of the filepath

    Returns:
        str: The content of the given file.
    """
    with open(filepath, "r") as f:
        data = f.read()
    return data

def get_token_pair_count(tokens: List[int], pairwise_count):
    """This method is used to get the word pair count accross the corpus

    Args:
        tokens (List[str]): The vocab containing all the tokens

    Returns:
        Dict[Tuple[str],int]: The contiguous word pairs and their counts
    """
    # pairwise_count = defaultdict(int)
    for i in range(len(tokens)-1):
        pairwise_count[(tokens[i],tokens[i+1])] += 1
    
    

def merge_token_pairs(tokens: List[int], top_pair: Tuple[str,str], new_idx:int) -> List[int]:
    """This method is used to merge tokens with the top tokens and return the new tokens after merging

    Args:
        tokens (List[int]): Initial set of tokens
        top_pair (Tuple[str,str]): This is the top pair of tokens which will be merged
        new_idx (int): This is the new index which will be assigned to the merged token

    Returns:
        List[int]: New set of tokens after merging
    """
    i = 0
    new_tokens = []
    while i<len(tokens):
        if i!=len(tokens)-1 and tokens[i]==top_pair[0] and tokens[i+1]==top_pair[1]:
            new_tokens.append(new_idx)
            i+=2
        else:
            new_tokens.append(tokens[i])
            i+=1
    return new_tokens

def replace_control_characters(s: str) -> str:
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)


def render_token(t: bytes) -> str:
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

def save_trained_tokenizer(pattern, vocab, merges, model_path = "../models/minGPT.model", vocab_path = "../models/minGPT.vocab"):
    
    model_file = model_path
    with open(model_file, 'w') as f:
        f.write("minbpe v1\n")
        f.write(f"{pattern}\n")
        for idx1, idx2 in merges:
            f.write(f"{idx1} {idx2}\n")
    
    vocab_file =  vocab_path
    inverted_merges = {idx: pair for pair, idx in merges.items()}
    with open(vocab_file, "w", encoding="utf-8") as f:
        for idx, token in vocab.items():

            s = render_token(token)
            if idx in inverted_merges:
                idx0, idx1 = inverted_merges[idx]
                s0 = render_token(vocab[idx0])
                s1 = render_token(vocab[idx1])
                f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
            else:
                f.write(f"[{s}] {idx}\n")
        
    print("Successfully saved the trained tokenizer")
        
def load_trained_tokenizer(fp = "../models/minGPT.model"):
    merges = {}
    special_tokens = {}
    idx = 256
    model_file = fp
    with open(model_file, 'r', encoding="utf-8") as f:
        # read the version
        version = f.readline().strip()
        assert version == "minbpe v1"
        # read the pattern
        pattern = f.readline().strip()
        # read the merges
        for line in f:
            idx1, idx2 = map(int, line.split())
            merges[(idx1, idx2)] = idx
            idx += 1

    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    
    return pattern, vocab, merges

        