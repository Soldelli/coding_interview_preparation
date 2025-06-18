import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(-math.log(10000.0) * torch.arange(0, d_model, 2).float() / d_model)

        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
    
def scaled_dot_product_attention(query, key, value, mask=None):
    '''
        query, key, value: [batch_size, num_heads, seq_len, d_k]
        mask: [batch_size, 1, seq_length, seq_length]
    '''
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)    # (batch_size, num_heads, seq_length, seq_length)

    if mask is not None:
        scores = scores.masked_fill(mask ==0, float('-inf'))  #[batch_size, 1, seq_length, seq_length]

    attn_weights = F.softmax(scores, dim=-1)    # (batch_size, num_heads, seq_length, seq_length)
    output = torch.matmul(attn_weights, value)  # (batch_size, num_heads, seq_length, d_k)
    return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads==0, 'd_model must be divisible by num_heads'
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # Project inputs
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # Reshape: (batch_size, seq_len, d_model) → (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)  
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)  
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)

        # Compute self attention
        attn_output, attn_weights = scaled_dot_product_attention(Q,K,V,mask=mask)

        # reshape outputs to d_model size
        attn_output = attn_output.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)

        # output_projection 
        attn_output = self.w_o(attn_output)
        return attn_output, attn_weights


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, factor=4, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.linear1 = nn.Linear(d_model,d_model*factor)
        self.linear2 = nn.Linear(d_model*factor, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.act(self.linear1(x))))
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, factor=4, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, factor, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        # apply attention
        attn_outout, _ = self.self_attn(x,x,x,mask=mask)
        x = self.norm1(x+self.dropout1(attn_outout))

        #apply ffn
        ffn_outout = self.ffn(x)
        x = self.norm2(x+self.dropout2(ffn_outout))
        return x
    

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, factor=4, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, factor, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        # self attention
        attn_output, _ =self.self_attn(x,x,x, mask=tgt_mask)
        x = self.norm1(x+self.dropout1(attn_output))

        # cross attention
        attn_output, _ =self.self_attn(x,enc_output,enc_output, mask=memory_mask)
        x = self.norm2(x+self.dropout2(attn_output))

        # feed forward
        ffn_output = self.ffn(x)
        x = self.norm3(x+self.dropout3(ffn_output))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, factor, dropout=0.1, max_len=5000):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, factor, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        x = self.embedding(src)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        x = self.norm(x)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, factor, dropout=0.1, max_len=5000):
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, factor, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, enc_output, tgt_mask=None, memory_mask=None):
        x = self.embedding(src)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, memory_mask)

        x = self.norm(x)
        return x
    

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_layers=6, num_heads=8, factor=4, dropout=0.1, max_len=5000):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, factor, dropout, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, factor, dropout, max_len)
        self.out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        logits = self.out(dec_output)
        return logits
    

if __name__ == "__main__":
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    batch_size = 8
    src_seq_len = 20
    tgt_seq_len = 20

    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))

    model = Transformer(src_vocab_size, tgt_vocab_size)
    logits = model(src, tgt)

    print(f'Output logit shape: {logits.shape}')
