import argparse
from collections import OrderedDict
import importlib
from RMLP_Model.my_utils import add_common_arguments

parser = argparse.ArgumentParser()
add_common_arguments(parser)
parser.add_argument('--deep_supervision', type=bool, help='deep supervision', default=False)
parser.add_argument('--data', type=str, help='Path to the data of decompensation task',
                    default='/home/thiti/Research_SGH/data/temp/mimic3benchmark/in-hospital-mortality')
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
parser.add_argument('--pattern_specs', type=str, help='pattern specs',
                    default='5-5')
parser.add_argument('--lstm_hidden_dim', type=int, help='lstm_hidden_dim', default='128')
parser.add_argument('--mlp_hidden_dim', type=int, help='mlp_hidden_dim', default='128')
parser.add_argument('--num_mlp_layers', type=int, help='num_mlp_layers', default='1')
parser.add_argument('--mlp_pattern_NN', type=str, help='mlp pattern specs from input side (left) to output side (right)', default='5-1')
parser.add_argument('--input_dim', type=int, help='input dimension', default='76')
parser.add_argument('--file_name', type=str, help='file_name for model',
                    default='trained_model')
parser.add_argument('--clip', type=float, help='gradient clipping', default='0.')
parser.add_argument('--target_repl_coef', type=float, help='target_repl_coef', default='0.5')
parser.add_argument('--gpu', type=str, help='Choose GPU', default='1')
parser.add_argument('--imputation', type=str, default='previous')
parser.add_argument('--log_likelihood_fn', type=str, help='log_likelihood_fn', default='./test')
args = parser.parse_args()
args.pattern_specs=OrderedDict(sorted(([int(y) for y in x.split("-")] for x in args.pattern_specs.split("_")),
                                key=lambda t: t[0]))
args.mlp_pattern_NN=[int(y) for y in args.mlp_pattern_NN.split("-")] 
print(args)

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

from RMLP_Model import my_utils

import numpy as np
import os
import imp
import re

from mimic3models import common_utils
from mimic3models.in_hospital_mortality import utils
from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models.preprocessing import Discretizer, Normalizer

import torch
seed=0
import random
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import numpy as np
np.random.seed(seed)

from RMLP_Model import RMLP_NN_mlpflex as Sopa_Decomp 

from mimic3models import metrics
from sklearn.metrics import log_loss
device = torch.device("cuda" if torch.cuda.is_available() == True else 'cpu')

target_repl = (args.target_repl_coef > 0.0)
# Build readers, discretizers, normalizers
train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                         listfile=os.path.join(args.data, 'train_listfile.csv'),
                                         period_length=48.0)

val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                       listfile=os.path.join(args.data, 'val_listfile.csv'),
                                       period_length=48.0)

test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                       listfile=os.path.join(args.data, 'test_listfile.csv'),
                                       period_length=48.0)

discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = 'ihm_ts{}.input_str_{}.start_time_zero.normalizer'.format(args.timestep, args.imputation)
normalizer_state = os.path.join(args.data, normalizer_state)
normalizer.load_params(normalizer_state)

# Read data
train_raw = utils.load_data(train_reader, discretizer, normalizer, args.small_part)
val_raw = utils.load_data(val_reader, discretizer, normalizer, args.small_part)
test_raw = utils.load_data(test_reader, discretizer, normalizer, args.small_part)

if target_repl:
    T = train_raw[0][0].shape[0]

    def extend_labels(data):
        data = list(data)
        labels = np.array(data[1])  # (B,)
        data[1] = [labels, None]
        data[1][1] = np.expand_dims(labels, axis=-1).repeat(T, axis=1)  # (B, T)
        data[1][1] = np.expand_dims(data[1][1], axis=-1)  # (B, T, 1)
        return data

    train_raw = extend_labels(train_raw)
    val_raw = extend_labels(val_raw)
    test_raw = extend_labels(test_raw)

# Generators
training_set = my_utils.Dataset_Target_Rep(train_raw)
training_generator = torch.utils.data.DataLoader(training_set,batch_size = args.batch_size , shuffle= True)

validation_set = my_utils.Dataset_Target_Rep(val_raw)
validation_generator = torch.utils.data.DataLoader(validation_set,batch_size = args.batch_size , shuffle= False)

#Model Creation
model=Sopa_Decomp.SoPa_MLP(input_dim=args.input_dim,
                            pattern_specs=args.pattern_specs,
                            semiring=Sopa_Decomp.LogSpaceMaxTimesSemiring,
                            mlp_hidden_dim=args.mlp_hidden_dim,
                            num_mlp_layers=args.num_mlp_layers,
                            num_classes=1,
                            mlp_pattern_NN=args.mlp_pattern_NN,
                            deep_supervision=args.deep_supervision,
                            gpu=True,
                            dropout=args.dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
if args.clip >0:
    my_utils.enable_gradient_clipping(model, clip=args.clip)

print('Start training ... ')

train_loss = []
val_loss = []
batch_loss = []
max_auprc = 0


file_name = './saved_weights/'+args.file_name
for each_chunk in range(args.epochs):
    cur_batch_loss = []
    model.train()
    each_batch=0
    for x_batch,yT_batch,yt_batch in training_generator:
        mask=torch.logical_not(torch.sum(x_batch==0,dim=2) == x_batch.shape[2]).to(device)
        x_batch=x_batch.transpose(1,2)

        optimizer.zero_grad()
        cur_output=model(x_batch,mask)
        yT_hat = cur_output[:,-1]
        yt_hat = cur_output
        loss_T=yT_batch * torch.log(yT_hat + 1e-7) + (1 - yT_batch) * torch.log(1 - yT_hat + 1e-7)
        loss_t=yt_batch * torch.log(yt_hat + 1e-7) + (1 - yt_batch) * torch.log(1 - yt_hat + 1e-7)
        loss_t = torch.mean(loss_t,dim=1)

        loss_T = torch.neg(torch.sum(loss_T,dim=0))
        loss_t = torch.neg(torch.sum(loss_t,dim=0))
        loss = (1-args.target_repl_coef)*loss_T + args.target_repl_coef*loss_t
        cur_batch_loss.append(loss.cpu().detach().numpy())
        
        
        loss.backward()
        optimizer.step()
        each_batch=each_batch+1        
        if each_batch % 50 == 0:
            print('Chunk %d, Batch %d: Loss = %.4f'%(each_chunk, each_batch, cur_batch_loss[-1]))
            
    batch_loss.append(cur_batch_loss)
    train_loss.append(np.mean(np.array(cur_batch_loss)))
    
    print("\n==>Predicting on validation")
    with torch.no_grad():
        model.eval()
        cur_val_loss = []
        valid_true = []
        valid_pred = []
        each_batch=0

        for valid_x,valid_yT,valid_yt in validation_generator:
            valid_mask=torch.logical_not(torch.sum(valid_x==0,dim=2) == valid_x.shape[2]).to(device)
            valid_x=valid_x.transpose(1,2)
            valid_output = model(valid_x,valid_mask)
            
            valid_output_T = valid_output[:,-1]
            valid_output_t = valid_output
            
            valid_loss_T = valid_yT * torch.log(valid_output_T + 1e-7) + (1 - valid_yT) * torch.log(1 - valid_output_T + 1e-7)
            valid_loss_t = valid_yt * torch.log(valid_output_t + 1e-7) + (1 - valid_yt) * torch.log(1 - valid_output_t + 1e-7)
            valid_loss_t = torch.mean(valid_loss_t,dim=1)
            
            valid_loss_T = torch.neg(torch.sum(valid_loss_T,dim=0))
            valid_loss_t = torch.neg(torch.sum(valid_loss_t,dim=0))
            valid_loss = (1-args.target_repl_coef)*valid_loss_T + args.target_repl_coef*valid_loss_t
            
            cur_val_loss.append(valid_loss.cpu().detach().numpy())
                  
            for t, p in zip(valid_yT.cpu().numpy().flatten(), valid_output_T.cpu().detach().numpy().flatten()):
                valid_true.append(t)
                valid_pred.append(p)
        
        val_loss.append(np.mean(np.array(cur_val_loss)))
        print('Valid loss = %.4f'%(val_loss[-1]))
        print('\n')
        valid_pred = np.array(valid_pred)
        valid_pred = np.stack([1 - valid_pred, valid_pred], axis=1)
        ret = metrics.print_metrics_binary(valid_true, valid_pred)
        print()
        
        cur_auprc = ret['auprc']
        cur_auroc = ret['auroc']
        if (cur_auprc > max_auprc) and (cur_auroc>=0.7):
            max_auprc = cur_auprc
            state = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'chunk': each_chunk
                }
            torch.save(state, file_name)
            print('\n------------ Save best model ------------\n')


# Load test data and prepare generators
test_set = my_utils.Dataset_Target_Rep(test_raw)
test_generator = torch.utils.data.DataLoader(test_set,batch_size = args.batch_size, shuffle= False)

'''Evaluate phase'''
print('Testing model ... ')

checkpoint = torch.load(file_name)
save_chunk = checkpoint['chunk']
print("last saved model is in chunk {}".format(save_chunk))
model.load_state_dict(checkpoint['net'])
optimizer.load_state_dict(checkpoint['optimizer'])
model.eval()


with torch.no_grad():
    cur_test_loss = []
    test_true = []
    test_pred = []

    for test_x,test_yT,test_yt in test_generator:
        test_mask=torch.logical_not(torch.sum(test_x==0,dim=2) == test_x.shape[2]).to(device)
        
        test_x=test_x.transpose(1,2)
        test_output = model(test_x,test_mask)
        
        test_output_T = test_output[:,-1]
        test_output_t = test_output
        test_loss_T = test_yT * torch.log(test_output_T + 1e-7) + (1 - test_yT) * torch.log(1 - test_output_T + 1e-7)
        test_loss_t = test_yt * torch.log(test_output_t + 1e-7) + (1 - test_yt) * torch.log(1 - test_output_t + 1e-7)
        test_loss_t = torch.mean(test_loss_t,dim=1)
            
        test_loss_T = torch.neg(torch.sum(test_loss_T,dim=0))
        test_loss_t = torch.neg(torch.sum(test_loss_t,dim=0))
        test_loss = (1-args.target_repl_coef)*test_loss_T + args.target_repl_coef*test_loss_t
        cur_test_loss.append(test_loss.cpu().detach().numpy())
        
        for t, p in zip(test_yT.cpu().numpy().flatten(), test_output_T.cpu().detach().numpy().flatten()):
            test_true.append(t)
            test_pred.append(p)
    def cross_entropy(predictions, targets):
        predictions=np.array(predictions)
        targets=np.array(targets)
        ce = -(targets * np.log(predictions) + (1-targets) * np.log(1-predictions) )
        return ce
            
    log_likelihood=cross_entropy(test_pred, test_true)
    log_likelihood_output=[neg_log for neg_log in log_likelihood]
    import pickle
    with open(args.log_likelihood_fn, "wb") as fp:   
        pickle.dump(log_likelihood_output, fp)

    print('Test neg log-likelihood = %.4f'%(log_loss(test_true, test_pred)))
    print('\n')
    test_pred = np.array(test_pred)
    test_pred = np.stack([1 - test_pred, test_pred], axis=1)
    test_ret = metrics.print_metrics_binary(test_true, test_pred)

    #Positive predictive value
    N_ppv=[ 100, 200, 300, 400, 374]
    pos_rate=np.zeros(len(N_ppv))
    pred_true=np.concatenate([test_pred[:,1].reshape(-1,1),np.array(test_true).reshape(-1,1)],axis=1)
    pred_true=pred_true[(-pred_true[:,0]).argsort()]
    for ppv in range(len(N_ppv)):
        No_Patients_Determined=N_ppv[ppv]
        pos_rate[ppv] = np.sum(pred_true[:No_Patients_Determined,1])/No_Patients_Determined

    print(f"For first highest n prediction \t - ppv is:")
    for n_ppv, pos in zip(N_ppv,pos_rate):
        print(f"\t {n_ppv} \t\t\t\t {pos}")

