import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype = torch.int64)  # since vocabulary could be bigger than 32bits , since SOS present in both lang we use anyone 
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype = torch.int64)  
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype = torch.int64)  

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        src_target_pair = self.ds[index]    
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids # ids number corresponding to each word in the original sentence
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids # input for the decoder 

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # since we have to remove for the SOS and EOS , (single sentence)
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # Right shifted so we only have SOS. 

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0 :
            raise ValueError('Sentence is too long')  # since seq_len should be greater than all sentence lengths.
        
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]* enc_num_padding_tokens, dtype=torch.int64)
            ],
            dim = 0
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, # (seq_len)
            "decoder_input": decoder_input, # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1,1, seq_len), first creates a boolean mask and then converts to int, so 0 at padding position and 1 at others
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),  # causal mask builds a matrix of size seq_len by seq-len (1, seq_len) & (1, seq_len, seq_len) this can be broadcasted
            "label": label, # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text
        }


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal = 1).type(torch.int) # returns the matrix with all lower values 0 including the diagonal, and we want that zero part only
    return mask == 0