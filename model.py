# language conversion
# super().__init__() is a way to call the constructor of the parent class to ensure proper initialization of inherited attributes and methods. It's a common practice when defining subclasses in Python.
# seq_len, is maximum length of the sentence the context window.
# register the tensor positional encoding as buffer to make it part of the model state_dict
# basically, when we have a tensor that we want to keep, no as a learned parameter, but want it to be saved and loaded with the model, we can register it as buffer. This way tensor will be saved along with state of the model.
# we also tell the model the positional encoding is not a learnable parameter, but it is part of the model state.
# eps for numerical stability and to avoid division by zero.
# alpha and bias are single values.
# does feed-forward layer goes for each token as each is encoded into d_model? does each token have same weights and bias ?
# Q,K,V exactly same in encoder but not so in decoder
# we multiply the attention step using @ and transpose but only seq-len ny d_k part because batch and h are fixed
# we dont want some words to not watch future words or we don't want padding values to not interact with other values, because these are just filler words to reach the sequence length
# contiguous needed for the transpose step, to transform the shape of a tensor we need the memory to be in contiguous  
# residual connection is basically the skip connector 
# module list is a way to organize the modules
# bias false part, also modulelist is decoderblock residual connection
# if we have two very different languages where sentence lenghts vary drastically, we could use different seq_len for source and target.
# tokenizer comes before input embeddings , there are BPE tokenizer, word level, sub-word level
# bias issue, log_softmax? in proj layer, errors in train.py, decode mask, config seq_len

import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding  = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        return self.embedding(x)* math.sqrt(self.d_model)
    


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape(seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model))
        # apply the sin to even position
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        # adding batch dimension
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as buffer 
        self.register_buffer('pe',pe)

    def forward(self,x):
        x = x + (self.pe[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)
        


class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # learnable parameter,hence nn.parameter, also it is multiplicative
        self.bias = nn.Parameter(torch.zeros(1)) # added learnable parameter

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)        # calculating the mean along the last dimension of the tensor, keep dims will keep the third dim so 1x seq_lenx d_model -> 1 x seq_len x 1 if keep dims not used last 1 wont be there.
        std = x.std(dim = -1, keepdim=True)    
        return self.alpha*(x - mean)/(std + self.eps) + self.bias      



class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1, input dim: d_model and output dim: d_ff
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2 , sicne bias is bydefault True

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch,seq_len,d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h:int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias= False)  
        self.w_k = nn.Linear(d_model, d_model, bias= False)
        self.w_v = nn.Linear(d_model, d_model, bias= False)
        self.w_o = nn.Linear(d_model, d_model, bias= False) # d_v = d_k
        self.dropout = nn.Dropout(dropout)

    # creating attention function to calculate, by using staticmethod we can call this function without an instance of the class, MultiHeadAttentionBlock.attention and use it 
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)   # (batch, h , seq_len, d_k) @ (batch, h, d_k, seq_len) = (batch, h, seq_len, seq_len)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) # (batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) -> (batch, h, seq_len, d_k) and , (batch, h, seq_len, seq_len)
        return (attention_scores @ value), attention_scores  # the tuple needed for the next layer but the attention_scores are used for visualization
    

    def forward(self, q, k, v, mask):
        query = self.w_q(q)   # (batch, seq_len, d_model) -->  (batch, seq_len, d_model) , Q'
        key = self.w_k(k)   # (batch, seq_len, d_model) -->  (batch, seq_len, d_model) 
        value = self.w_v(v) # (batch, seq_len, d_model) -->  (batch, seq_len, d_model) 

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        # transpose because we want h to be the 2nd dimension, this way all heads will see the whole sentence seq_len by d_k
        query  = query.view(query.shape[0],query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h*self.d_k)

        # multiply by w_o
        return self.w_o(x)  # (batch, seq_len, d_model)

class ResidualConnection(nn.Module):

    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm  = LayerNormalization()

    def forward(self, x, sublayer):  # sublayer is the previous layer
        return x + self.dropout(sublayer(self.norm(x)))  # there is slight differene, here we apply the norm then apply the sublayer and add it to the originial , but in paper we first  pass by sublayer then norm
    

class EncoderBlock(nn.Module):


    def __init__(self, self_attention_block: MultiHeadAttentionBlock , feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)]) # since we need two residual connections


    def forward(self, x, src_mask):  # src_mask is the mask we want to apply to the input of the encoder, to hide the interaction of the padding words with other words
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:  # since we have n layers and all are applied one after another
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()  # after the final N encoder blocks 


    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)
    


class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock,dropout: float) -> None:
        super().__init__()
        self.self_attetnion_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):  # src_mask : for the encoder and for original language , tgt_mask: for the deocder and for the output language
        x = self.residual_connections[0](x, lambda x: self.self_attetnion_block(x,x,x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()


    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    

class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_zize)
        # return torch.log_softmax(self.proj(x), dim = -1)   # using log for numerical stability
        return self.proj(x)
     
    

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:   # since we are dealing with multiple languages , we need source embedding and targer embeddings
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer =  projection_layer

    # we define 3 methods, one to encode , one to decode and one to project, we are not using a single forward function because we want to use the encoder computations during inference, also helps in visualization of the atttention


    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask,tgt ,tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self,x):
        return self.projection_layer(x)
    

# here we are using for translation but could be used for other use cases also. so, naming are for translation tasks
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8,dropout: float=0.1, d_ff: int =2048) -> Transformer:
    
    # Create the Embedding Layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Positional encoding, (one would do as same and only depends on position)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block,feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialization the parameters using xavier uniform initialization
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer