from .interface import CLIPInterface
from clip.simple_tokenizer import SimpleTokenizer as _tokenizer

SimpleTokenizer = _tokenizer()