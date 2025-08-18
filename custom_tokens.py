import sys
import pandas as pd
from typing import List, Set, Union
from nltk.tokenize import RegexpTokenizer
from collections import Counter

def custom_tokenizer(txt: str, spec_tokens: List[str], pattern: str = r"|\d|\w+|[^\s]") -> List[str]:
    """
    Tokenize text into words or characters using NLTK's RegexpTokenizer, considerung 
    given special combinations as single tokens.

    :param txt: The corpus as a single string element.
    :param spec_tokens: A list of special tokens (e.g. ending, out-of-vocab).
    :param pattern: By default the corpus is tokenized on a word level (split by spaces).
                    Numbers are considered single tokens.

    >> note: The pattern for character level tokenization is '|.'
    """
    pattern = "|".join(spec_tokens) + pattern
    tokenizer = RegexpTokenizer(pattern)
    tokens = tokenizer.tokenize(txt)
    return tokens


def get_infrequent_tokens(tokens: Union[List[str], str], min_count: int) -> List[str]:
    """
    Identify tokens that appear less than a minimum count.
    
    :param tokens: When it is the raw text in a string, frequencies are counted on character level.
                   When it is the tokenized corpus as list, frequencies are counted on token level.
    :min_count: Threshold of occurence to flag a token.
    :return: List of tokens that appear infrequently. 
    """
    counts = Counter(tokens)
    infreq_tokens = set([k for k,v in counts.items() if v<=min_count])
    return infreq_tokens

def mask_tokens(tokens: List[str], mask: Set[str]) -> List[str]:
    """
    Iterate through all tokens. Any token that is part of the set, is replaced by the unknown token.

    :param tokens: The tokenized corpus.
    :param mask: Set of tokens that shall be masked in the corpus.
    :return: List of tokenized corpus after the masking operation.
    """
    unknown_token = "<UNK>"
    return [t.replace(t, unknown_token) if t in mask else t for t in tokens]

