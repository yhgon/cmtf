from cmtf import CompressiveTransformer
from cmtf_ar_wrapper import AutoregressiveWrapper

import argparse

import random
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# constants
def parse_args(parser):
    """
    Parse commandline arguments.
    """
    build_model = parser.add_argument_group('model setup')    
    build_model.add_argument( '--num_tokens',      type=int,  default=256, help='')
    build_model.add_argument( '--dim',             type=int,  default=512, help='')
    build_model.add_argument( '--depth',           type=int,  default=  8, help='')
    build_model.add_argument( '--heads',           type=int,  default=  8, help='')       
    build_model.add_argument( '--seq_len',         type=int,  default=512, help='')   
    build_model.add_argument( '--mem_len',         type=int,  default=512, help='')   
    build_model.add_argument( '--cmem_len',        type=int,  default=128, help='')           
    build_model.add_argument( '--num_mem_layers',  type=int,  default=  3, help='')     

    training = parser.add_argument_group('training setup')    

    training.add_argument( '--validate_every',  type=int,  default=100, help='')
    training.add_argument( '--generate_every',  type=int,  default=200, help='')
    training.add_argument( '--prime_length',    type=int,  default= 512, help='')
    training.add_argument( '--generate_length', type=int,  default=1024, help='')    


    optimization = parser.add_argument_group('optimization setup')
    optimization.add_argument( '--optimizer',      type=str,  default='adam',   help='Optimization algorithm')
    optimization.add_argument( '--learning_rate',  type=int,  default=1e-4,     help='learning rate')
    optimization.add_argument( '--num_batches',    type=int,  default=100000,   help='max iteration')
    optimization.add_argument( '--batch_size',     type=int,  default=16,       help='batch size')    
    optimization.add_argument( '--max_batch_size', type=int,  default=4,        help='gradient accumulation')    


    dataset = parser.add_argument_group('dataset parameters')
    dataset.add_argument('--zip_filename',  type=str, default='/content/enwik8.gz',  help='Path to training filelist')
    dataset.add_argument('--num_segments',  type=int, default =   4,  help='num_segments')        
     
    return parser   


# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))


# prepare enwik8 data
def prepare_dataset(zip_filename):
    with gzip.open(zip_filename) as file:
        X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
        trX, vaX = np.split(X, [int(90e6)])
        data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)
    return data_train, data_val 


class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len, segments):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.segments = segments
        self.total_len = seq_len * segments

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.total_len - 1, (1,))
        full_seq = self.data[rand_start: rand_start + self.total_len + 1].long()
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.total_len


# training

def run(args):    


    # build model 
    print("prepare model")
    model = CompressiveTransformer(
        num_tokens = args.num_tokens,
        dim = args.dim,
        depth = args.depth,
        seq_len = args.seq_len,
        mem_len = args.mem_len,
        cmem_len = args.cmem_len,
        heads = args.heads,
        memory_layers = [*range(args.depth-3+1,args.depth+1,1)]
    )

    model = AutoregressiveWrapper(model)
    model.cuda()

    # prepare dataset
    print("prepare dataset")
    data_train, data_val  = prepare_dataset(args.zip_filename)

    train_dataset = TextSamplerDataset(data_train, args.seq_len , args.num_segments)
    val_dataset   = TextSamplerDataset(data_val, args.seq_len , args.num_segments)

    train_loader  = cycle(DataLoader(train_dataset, batch_size = args.batch_size))
    val_loader    = cycle(DataLoader(val_dataset, batch_size = args.batch_size))

    # optimizer
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


    print("start loop")
    for i in tqdm.tqdm(range(args.num_batches), mininterval=10., desc='training'):
        model.train()

        grad_accum_every = args.batch_size / args.max_batch_size

        for mlm_loss, aux_loss, is_last in model(next(train_loader), max_batch_size = args.max_batch_size, return_loss = True):
            loss = mlm_loss + aux_loss
            (loss / grad_accum_every).backward()

            print(f' {i:d} training loss: {mlm_loss.item():.4f} | aux_loss: {aux_loss.item():.4f}')

            if is_last:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optim.step()
                optim.zero_grad()

        if i % args.validate_every == 0:
            model.eval()
            with torch.no_grad():
                for loss, aux_loss, _ in model(next(val_loader), return_loss = True):
                    print(f'validation loss: {loss.item():.4f}')

        if i % args.generate_every == 0:
            model.eval()
            inp = random.choice(val_dataset)[:-1]
            inp = inp[:args.prime_length]
            prime = decode_tokens(inp)
            print(f'%s \n\n %s', (prime, '*' * 100))

            sample = model.generate(inp, args.generate_length)
            output_str = decode_tokens(sample)
            print(output_str)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PyTorch WaveGrad Training', 
                                     allow_abbrev=False) 
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    print(args)   
    ### additional configuration from config file 
    #with open(args.config) as f:
    #    config = ConfigWrapper(**json.load(f))
                

    
    run( args)


