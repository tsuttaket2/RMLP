import torch
import numpy as np
import argparse
from argparse import ArgumentParser
import os 
from .data_sst import Vocab, UNK_IDX, START_TOKEN_IDX, END_TOKEN_IDX
from .util import chunked_sorted, to_cuda, right_pad
from torch.autograd import Variable
from torch import FloatTensor

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def fixed_var(tensor):
    return Variable(tensor, requires_grad=False)

def y_mask_sdoc_decomp(batch_ts,batch_y,seq_length):
    device = torch.device("cuda" if torch.cuda.is_available() == True else 'cpu')
    id=0
    batch_mask=torch.zeros((len(batch_ts),1,1),dtype=torch.int64).to(device)
    for ts in batch_ts:
        batch_mask[id]=int(ts[-1]-1)
        id+=1
    batch_y=torch.gather(batch_y,1,batch_mask)
    batch_mask = torch.logical_not(torch.arange(seq_length).unsqueeze(0).to(device) > batch_mask.squeeze(1))
    return (batch_y,batch_mask)

def mask_st_decomp(batch_ts,seq_length):
    device = torch.device("cuda" if torch.cuda.is_available() == True else 'cpu')
    id=0
    batch_mask=torch.zeros((len(batch_ts),seq_length,1),dtype=torch.bool).to(device)
    for ts in batch_ts:
        batch_mask[id,int(ts[-1]-1),0]=True
        id+=1
    return batch_mask

class Dataset(torch.utils.data.Dataset):
    def __init__(self, raw_data):
        self.device = torch.device("cuda" if torch.cuda.is_available() == True else 'cpu')
        X = raw_data[0].astype('float32')
        Y = np.array(raw_data[1]).astype('float32')
        X = torch.from_numpy(X).to(self.device)
        Y=torch.from_numpy(Y).to(self.device)
        
        'Initialization'
        self.X = X  #Batch Size x Time (48) x Input_dim
        self.Y = Y

    def __len__(self):
        'Denotes the total number of samples'
        return self.X.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.X[index,], self.Y[index]

class Dataset_Target_Rep(torch.utils.data.Dataset):
    def __init__(self, raw_data):
        self.device = torch.device("cuda" if torch.cuda.is_available() == True else 'cpu')
        X = raw_data[0].astype('float32')
        YT = np.array(raw_data[1][0]).astype('float32')
        Yt = np.array(raw_data[1][1]).astype('float32')
        X = torch.from_numpy(X).to(self.device)
        YT = torch.from_numpy(YT).to(self.device)
        Yt = torch.from_numpy(Yt).to(self.device)
        
        'Initialization'
        self.X = X  #Batch Size x Time (48) x Input_dim
        self.YT = YT
        self.Yt = Yt

    def __len__(self):
        'Denotes the total number of samples'
        return self.X.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.X[index,], self.YT[index], self.Yt[index].squeeze(-1)
        
def enable_gradient_clipping(model, clip) -> None:
    if clip is not None and clip > 0:
        # Pylint is unable to tell that we're in the case that _grad_clipping is not None...
        # pylint: disable=invalid-unary-operand-type
        clip_function = lambda grad: grad.clamp(-clip, clip)
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.register_hook(clip_function)

def add_common_arguments(parser):
    """ Add all the parameters which are common across the tasks
    """
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of chunks to train')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--timestep', type=float, default=1.0)
    parser.add_argument('--small_part', dest='small_part', action='store_true')
    
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    
    parser.set_defaults(small_part=False)


def soft_pattern_arg_parser():
    """ CLI args related to SoftPatternsClassifier """
    p = ArgumentParser(add_help=False)
    p.add_argument("-p", "--patterns",
                   help="Pattern lengths and numbers: an underscore separated list of length-number pairs",
                   default="5-50_4-50_3-50_2-50")
    p.add_argument("-d", "--mlp_hidden_dim", help="MLP hidden dimension", type=int, default=25)
    p.add_argument("-y", "--num_mlp_layers", help="Number of MLP layers", type=int, default=2)

    return p




def training_arg_parser():
    """ CLI args related to training models. """
    p = ArgumentParser(add_help=False)
    p.add_argument("-i", "--num_iterations", help="Number of iterations", type=int, default=10)
    p.add_argument("--patience", help="Patience parameter (for early stopping)", type=int, default=30)
    p.add_argument("-m", "--model_save_dir", help="where to save the trained model")
    p.add_argument("-r", "--scheduler", help="Use reduce learning rate on plateau schedule", action='store_true')
    p.add_argument("--td", help="Train data file", required=True)
    p.add_argument("--tl", help="Train labels file", required=True)
    p.add_argument("--pre_computed_patterns", help="File containing pre-computed patterns")
    p.add_argument("-l", "--learning_rate", help="Adam Learning rate", type=float, default=1e-3)
    p.add_argument("--clip", help="Gradient clipping", type=float, default=None)
    return p

def general_arg_parser():
    """ CLI args related to training and testing models. """
    p = ArgumentParser(add_help=False)
    p.add_argument("-b", "--batch_size", help="Batch size", type=int, default=1)
    p.add_argument("--max_doc_len",
                   help="Maximum doc length. For longer documents, spans of length max_doc_len will be randomly "
                        "selected each iteration (-1 means no restriction)",
                   type=int, default=-1)
    p.add_argument("-s", "--seed", help="Random seed", type=int, default=100)
    p.add_argument("--vd", help="Validation data file", required=True)
    p.add_argument("--vl", help="Validation labels file", required=True)
    p.add_argument("--testd", help="Test data file", required=True)
    p.add_argument("--testl", help="Test labels file", required=True)
    p.add_argument("--input_model", help="Input model (to run test and not train)")
    p.add_argument("-t", "--dropout", help="Use dropout", type=float, default=0)
    p.add_argument("-g", "--gpu", help="Use GPU", action='store_true')
    p.add_argument("-e", "--embedding_file", help="Word embedding file", required=True)

    return p

def mask_sst(docs_len, batch_mask):
    for i,doc_len in enumerate(docs_len):
        batch_mask[i,:doc_len]=1
    return batch_mask


def argmax(output):
    """ only works for kxn tensors """
    _, am = torch.max(output, 1)
    return am

class Batch:
    """
    A batch of documents.
    Handles truncating documents to `max_len`, looking up word embeddings,
    and padding so that all docs in the batch have the same length.
    Makes a smaller vocab and embeddings matrix, only including words that are in the batch.
    """
    def __init__(self, docs, embeddings, cuda, word_dropout=0, max_len=-1):
        # print(docs)
        mini_vocab = Vocab.from_docs(docs, default=UNK_IDX, start=START_TOKEN_IDX, end=END_TOKEN_IDX)
        # Limit maximum document length (for efficiency reasons).
        if max_len != -1:
            docs = [doc[:max_len] for doc in docs]
        doc_lens = [len(doc) for doc in docs]
        self.doc_lens = cuda(torch.LongTensor(doc_lens))
        self.max_doc_len = max(doc_lens)
        if word_dropout:
            # for each token, with probability `word_dropout`, replace word index with UNK_IDX.
            docs = [
                [UNK_IDX if np.random.rand() < word_dropout else x for x in doc]
                for doc in docs
            ]
        # pad docs so they all have the same length.
        # we pad with UNK, whose embedding is 0, so it doesn't mess up sums or averages.
        docs = [right_pad(mini_vocab.numberize(doc), self.max_doc_len, UNK_IDX) for doc in docs]
        self.docs = [cuda(fixed_var(torch.LongTensor(doc))) for doc in docs]
        local_embeddings = [embeddings[i] for i in mini_vocab.names]
        self.embeddings_matrix = cuda(fixed_var(FloatTensor(local_embeddings).t()))

    def size(self):
        return len(self.docs)

def remove_old(old_reduced_model_path):
    if old_reduced_model_path != "" and os.path.isfile(old_reduced_model_path):
        os.remove(old_reduced_model_path)

def to_file(new_model, new_pattern_specs, logging_dir_filename):
    #new_model, new_pattern_specs = extract_learned_structure(model, args)

    if new_model is None:
        return new_pattern_specs, ""

    reduced_model_path = get_model_filepath(logging_dir_filename, new_pattern_specs)
    print("Writing model to", reduced_model_path)
    torch.save(new_model.state_dict(), reduced_model_path)

    return new_pattern_specs, reduced_model_path

def get_model_filepath(logging_dir_filename, pattern_specs):
    
    reduced_model_path = logging_dir_filename 
    reduced_model_path += "_model_learned={}.pth".format(pattern_specs)
    return reduced_model_path

def chunked(xs, chunk_size):
    """ Splits a list into `chunk_size`-sized pieces. """
    xs = list(xs)
    return [
        xs[i:i + chunk_size]
        for i in range(0, len(xs), chunk_size)
    ]


def decreasing_length(xs):
    return sorted(list(xs), key=lambda x: len(x[0]), reverse=True)


def chunked_sorted(xs, chunk_size):
    return chunked(decreasing_length(xs), chunk_size)


def shuffled_chunked_sorted(xs, chunk_size):
    """ Splits a list into `chunk_size`-sized pieces. """
    chunks = chunked_sorted(xs, chunk_size)
    np.random.shuffle(chunks)
    return chunks