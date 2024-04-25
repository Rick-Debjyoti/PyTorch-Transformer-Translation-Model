import torch 
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from dataset import BilingualDataset, causal_mask
from model import build_transformer
# import torchtext.datasets as datasets
#from torch.optim.lr_scheduler import LambdaLR
from config import get_weights_file_path, get_config, latest_weights_file_path
import warnings

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel                  
from tokenizers.trainers import WordLevelTrainer  # train on the sentences to get to the vocab size
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter
from torchmetrics.text import WordErrorRate, CharErrorRate, BLEUScore
from tqdm import tqdm

from pathlib import Path  # helps create absolute path and relative paths
# we pretend a configuration  called the tokenizer_file, which is the path to the tokenizer file and this path is formattable using the lang


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt,max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every token, we get from the decoder
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder 
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)   # only SOS as initializer for decoder same type as source, (1,1) one for batch, one for tokens of the data 
    while True:  # we will keep asking decoder for next output, until we get EOS or max_len is reached
        if decoder_input.size(1) == max_len:
            break
        #Build mask for the target (decoder_input)
        # We dont need another mask, since no need for  padding token
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:,-1])
        _, next_word = torch.max(prob, dim=1)   # Greedy Search
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)



# for us to see the output of the model
def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples = 2):
    model.eval()
    count = 0
    
    source_texts = []
    expected = []
    predicted = []

    # Size of the control window (just use a default value)
    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():  # we just want to inference from the model, we don't want to train it 
        for batch in validation_ds:  # for val_ds we have a tach size 1 
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Print to the console 
            print_msg('-'* console_width)
            print_msg(f'SOURCE: {source_text}')
            print_msg(f'TARGET: {target_text}')
            print_msg(f'PREDICTED: {model_out_text}')

            if count == num_examples:
                print_msg('-'*console_width)
                break


    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()




def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]



# method to build the tokenizer
def get_or_build_tokenizer(config, ds, lang):
    # config['tokenizer_file'] = '../tokenizers/tokenizer_{0}.json'
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token = "[UNK]")) # if our tokenizer sees that it doesn't recognize in the vocabulary it will replace 
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]","[SOS]","[EOS]"], min_frequency=2)  # for a word to appear in our vocabulary, it should be present atleast twice
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer



def get_ds(config):  # config of the model , which we will define later
    
    #ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    ds_raw = load_dataset('hind_encorp', split='train')   #  'hind_encorp' , "cfilt/iitb-english-hindi"

    # Build tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Keep 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size =1, shuffle=True)  # 1 because we want to process each sentence one by one

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt




def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model



def train_model(config):
    # Define the device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)
 
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps= 1e-9)

    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')



    loss_fn = nn.CrossEntropyLoss(ignore_index= tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device) # we don't want the padding token to contribute to the loss, label_smoothing helps model to be less confident about its decision, wetake 0.1 percent of probability of the predicted token and distribute among other tokens, helps reduce over-fitting

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:
            #model.train()    ##############################
            encoder_input = batch['encoder_input'].to(device)  #  (batch, seq_len)
            decoder_input = batch['decoder_input'].to(device)  #  (batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch, 1,1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (batch, 1, seq_len, seq_len)

            # run the tensor through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask)  # (batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)  # (batch, seq_len, d_model)
            proj_output = model.project(decoder_output)  # (batch, seq_len, tgt_vocab_size)

            label = batch['label'].to(device) # (batch, seq_len)

            # (batch, seq_len, tgt_vocab_size) --> (batch * seq_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # log the loss to tensorboard
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


            #run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
            global_step += 1
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of each epoch 
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch, 
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    config = get_config()
    train_model(config)