import numpy as np

from torch import FloatTensor, LongTensor, cat, mm, norm, randn, zeros, ones
from torch.nn import Parameter
from torch.nn import Module
from torch.autograd import Variable
from torch.nn.functional import sigmoid, log_softmax, tanh
import torch
from mimic3models import metrics
from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SoPa_MLP(Module):
    def __init__(self,
                 input_dim,
                 pattern_specs,
                 semiring,
                 mlp_hidden_dim,
                 num_mlp_layers,
                 num_classes,
                 mlp_pattern_NN,
                 deep_supervision=False,
                 gpu=True,
                 dropout=0.4):
        super(SoPa_MLP,self).__init__()

        self.deep_supervision=deep_supervision
        self.total_num_patterns = sum(pattern_specs.values())
        
        self.sopa = SoftPatternClassifier(input_dim,pattern_specs,semiring, mlp_pattern_NN,deep_supervision,gpu,dropout)
        self.mlp = MLP(self.total_num_patterns, mlp_hidden_dim, num_mlp_layers, num_classes)

        self.to_cuda =self.sopa.to_cuda
        self.dropout=dropout
        self.mlp_hidden_dim = mlp_hidden_dim
        self.num_mlp_layers = num_mlp_layers
        self.num_classes = num_classes
    def forward(self,input,mask):

        if self.deep_supervision :
            s_t=self.sopa(input,mask,self.dropout)
            out=self.to_cuda(torch.zeros((s_t.shape[0],s_t.shape[2])))
            for t in range(s_t.shape[2]):
                out[:,t]=self.mlp(s_t[...,t]).squeeze(-1)
            #Output will be Batch_Size x Prediction x Time Length
            return out
        else :
            s_doc = self.sopa(input,mask,self.dropout)
            return (self.mlp(s_doc)).squeeze(-1)

        

def identity(x):
    return x
def neg_infinity(*sizes):
    return -100 * ones(*sizes)  # not really -inf, shh
def to_cuda(gpu):
    return (lambda v: v.cuda()) if gpu else identity
def fixed_var(tensor):
    return Variable(tensor, requires_grad=False)
def normalize(data):
    length = data.shape[0]
    for i in range(length):
        data[i] = data[i] / norm(data[i])  # unit length

class Semiring:
    def __init__(self,
                 zero,
                 one,
                 plus,
                 times,
                 from_float,
                 to_float):
        self.zero = zero
        self.one = one
        self.plus = plus
        self.times = times
        self.from_float = from_float
        self.to_float = to_float
MaxPlusSemiring = \
    Semiring(
        neg_infinity,
        zeros,
        torch.max,
        torch.add,
        identity,
        identity
    )
ProbSemiring = \
    Semiring(
        zeros,
        ones,
        torch.add,
        torch.mul,
        sigmoid,
        identity
    )
ProbSemiring2 = \
    Semiring(
        zeros,
        ones,
        torch.add,
        torch.mul,
        tanh,
        identity
    )
# element-wise max, times. in log-space
LogSpaceMaxTimesSemiring = \
    Semiring(
        neg_infinity,
        zeros,
        torch.max,
        torch.add,
        lambda x: torch.log(torch.min(torch.sigmoid(x)+1e-7,torch.tensor([1.]).to(device))),
        torch.exp
    )


#Neural Network
def custom_model(data,list_static_models):
    #data batch_size x input_dim x time_seq
    #list_static_models n_models
    data=data.transpose(1,2)
    out_list=[]
    for m in list_static_models:
        out=m(data)

        out_list.append(out)
        
    #output batch_size x time_seq x n_models
    output=torch.cat(out_list,2)
    return output

class SoftPatternClassifier(Module):
    """
    A text classification model that feeds the document scores from a bunch of
    soft patterns into an MLP
    """
    def __init__(self,
                 input_dim,
                 pattern_specs,
                 semiring,
                 NN_mlp_pattern,
                 deep_supervision=False,
                 gpu=True,
                 dropout=0.4):
        super(SoftPatternClassifier, self).__init__()
        self.to_cuda = to_cuda(gpu)

        self.pattern_specs = pattern_specs
        self.max_pattern_length = max(list(pattern_specs.keys()))
        
        self.semiring=semiring
        self.input_dim=input_dim
        self.deep_supervision = deep_supervision
        end_states = [
            [end]
            for pattern_len, num_patterns in self.pattern_specs.items()
            for end in num_patterns * [pattern_len - 1]
        ]

        self.end_states = self.to_cuda(fixed_var(LongTensor(end_states)))

        self.num_diags=2
        self.total_num_patterns = sum(pattern_specs.values())
        self.NN_mlp_pattern = NN_mlp_pattern

        diag_data_size = self.total_num_patterns * self.num_diags * self.max_pattern_length

        self.list_static_models = torch.nn.ModuleList([self.to_cuda(MLP_noactivation(self.input_dim, self.NN_mlp_pattern, 1)) for _ in range(diag_data_size)])
        
        self.epsilon = Parameter(randn(self.total_num_patterns, self.max_pattern_length - 1))

        self.epsilon_scale = self.to_cuda(fixed_var(semiring.one(1)))
        
        self.dropout = torch.nn.Dropout(dropout)
        print("# params:", sum(p.nelement() for p in self.parameters()))
        
    def get_transition_matrices(self,batch,dropout):
        b = batch.size()[0]
        n = batch.size()[2]
        assert self.input_dim==batch.size()[1]
        
        transition_scores = \
            self.semiring.from_float(custom_model(batch,self.list_static_models) )
            
        if dropout is not None and dropout:
            transition_scores = self.dropout(transition_scores)
        
        
        batched_transition_scores = transition_scores.view(
            b, n, self.total_num_patterns, self.num_diags, self.max_pattern_length)
        return batched_transition_scores.transpose(0,1)
    def get_eps_value(self):
        return self.semiring.times(
            self.epsilon_scale,
            self.semiring.from_float(self.epsilon)
        )
    def forward(self, batch, mask, dropout=None):
        """ Calculate scores for one batch of documents. """
        transition_matrices = self.get_transition_matrices(batch,dropout)
        
        
        batch_size = batch.size()[0]
        num_patterns = self.total_num_patterns
        n = batch.size()[2]
        doc_lens=torch.sum(mask,dim=1)

        scores = self.to_cuda(fixed_var(self.semiring.zero((batch_size, num_patterns))))
        
        # to add start state for each word in the document.
        restart_padding = self.to_cuda(fixed_var(self.semiring.one(batch_size, num_patterns, 1)))

        zero_padding = self.to_cuda(fixed_var(self.semiring.zero(batch_size, num_patterns, 1)))

        eps_value = self.get_eps_value()

        batch_end_state_idxs = self.end_states.expand(batch_size, num_patterns, 1)
        hiddens = self.to_cuda(Variable(self.semiring.zero(batch_size,
                                                           num_patterns,
                                                           self.max_pattern_length)))
        if self.deep_supervision:
            s_t = self.to_cuda(self.semiring.zero(batch_size, num_patterns,n))
        
        # set start state (0) to 1 for each pattern in each doc
        hiddens[:, :, 0] = self.to_cuda(self.semiring.one(batch_size, num_patterns))
        
            
        for i, transition_matrix in enumerate(transition_matrices):
            self_loop_scale=None
            hiddens = self.transition_once(eps_value,
                                           hiddens,
                                           transition_matrix,
                                           zero_padding,
                                           restart_padding,
                                           self_loop_scale)
            
            # Look at the end state for each pattern, and "add" it into score
            end_state_vals = torch.gather(hiddens, 2, batch_end_state_idxs).view(batch_size, num_patterns)  #Equation8a
            if self.deep_supervision:
                s_t[...,i]=self.semiring.to_float(end_state_vals) #Equation8b
      
            # but only update score when we're not already past the end of the doc
            active_doc_idxs = torch.nonzero(torch.gt(doc_lens, i)).squeeze()

            scores[active_doc_idxs] = \
                self.semiring.plus(
                    scores[active_doc_idxs].clone(),
                    end_state_vals[active_doc_idxs]
                )
        
        scores = self.semiring.to_float(scores)
        if self.deep_supervision :
            return s_t
        else:
            return scores
    def transition_once(self,
                        eps_value,
                        hiddens,
                        transition_matrix_val,
                        zero_padding,
                        restart_padding,
                        self_loop_scale):
        after_epsilons = self.semiring.plus(hiddens,cat((zero_padding,self.semiring.times(hiddens[:, :, :-1],eps_value)), 2))
        
        after_main_paths = \
            cat((restart_padding,  # <- Adding the start state
                 self.semiring.times(
                     after_epsilons[:, :, :-1],
                     transition_matrix_val[:, :, -1, :-1])
                 ), 2)
        
        after_self_loops = \
                self.semiring.times(
                    after_epsilons,
                    transition_matrix_val[:, :, 0, :]
                )
        return self.semiring.plus(after_main_paths, after_self_loops)

from argparse import ArgumentParser

from torch.nn import Linear, Module, ModuleList
from torch.nn.functional import relu


class MLP(Module):
    """
    A multilayer perceptron with one hidden ReLU layer.
    Expects an input tensor of size (batch_size, input_dim) and returns
    a tensor of size (batch_size, output_dim).
    """
    def __init__(self,
                 input_dim,
                 hidden_layer_dim,
                 num_layers,
                 num_classes):
        super(MLP, self).__init__()

        self.num_layers = num_layers

        # create a list of layers of size num_layers
        layers = []
        for i in range(num_layers):
            d1 = input_dim if i == 0 else hidden_layer_dim
            d2 = hidden_layer_dim if i < (num_layers - 1) else num_classes
            layer = Linear(d1, d2)
            layers.append(layer)

        self.layers = ModuleList(layers)

    def forward(self, x):
        res = self.layers[0](x)
        for i in range(1, len(self.layers)):
            res = self.layers[i](relu(res))
        res=torch.sigmoid(res)
        return res

class MLP_noactivation(Module):
    """
    A multilayer perceptron with one hidden ReLU layer.
    Expects an input tensor of size (batch_size, input_dim) and returns
    a tensor of size (batch_size, output_dim).
    """
    def __init__(self,
                 input_dim,
                 mlp_pattern,
                 num_classes):
        super(MLP_noactivation, self).__init__()

        self.num_layers = len(mlp_pattern)
        self.leaky_relu=torch.nn.LeakyReLU(0.1)

        # create a list of layers of size num_layers
        layers = []
        for i in range(self.num_layers+1):
            d1 = input_dim if i == 0 else mlp_pattern[i-1]
            d2 = mlp_pattern[i] if i < self.num_layers else num_classes
            layer = Linear(d1, d2)
            layers.append(layer)

        self.layers = ModuleList(layers)

    def forward(self, x):
        res = self.layers[0](x)
        for i in range(1, len(self.layers)):
            res = self.layers[i](self.leaky_relu(res))
        res=res
        return res

THRESHOLD=0.1
def extract_learned_structure(model):
    regularization_groups = get_regularization_groups(model.sopa)
    end_states = [ [end] for pattern_len, num_patterns in model.sopa.pattern_specs.items()
                            for end in num_patterns * [pattern_len - 1] ]
    assert len(end_states) == len(regularization_groups)
    
    #Boolean tensor telling which pattern should be deleted
    WFSA_delete = regularization_groups < THRESHOLD
    new_num_states=[e[0]+1 for wfsa,e in zip(WFSA_delete,end_states) if torch.logical_not(wfsa) ]
    new_states=set(new_num_states)
    new_pattern_specs = OrderedDict.fromkeys(sorted(new_states),0)
    if new_pattern_specs:

        for i in new_num_states:
            new_pattern_specs[i]+=1
        Last_layer=model.mlp.layers[model.mlp.num_layers-1]
        
        new_model = SoPa_MLP(input_dim=76,
                            pattern_specs=new_pattern_specs,
                            semiring=model.sopa.semiring ,
                            mlp_hidden_dim=model.mlp_hidden_dim,
                            num_mlp_layers=model.num_mlp_layers,
                            num_classes=1,
                            deep_supervision=model.deep_supervision,
                            gpu=True,
                            dropout=model.dropout).to(device)
        #New model parameters
        new_weights=new_model.sopa.diags.reshape(new_model.sopa.total_num_patterns , new_model.sopa.num_diags , new_model.sopa.max_pattern_length,new_model.sopa.diags.shape[1])
        new_biases=new_model.sopa.bias.reshape(new_model.sopa.total_num_patterns , new_model.sopa.num_diags , new_model.sopa.max_pattern_length,1)
        new_epsilon=new_model.sopa.epsilon
        #Old model parameters
        old_weights= model.sopa.diags.reshape(model.sopa.total_num_patterns , model.sopa.num_diags , model.sopa.max_pattern_length,model.sopa.diags.shape[1])
        old_biases= model.sopa.bias.reshape(model.sopa.total_num_patterns , model.sopa.num_diags , model.sopa.max_pattern_length,1)
        old_epsilon=model.sopa.epsilon
        
        #Copy the params values from old to new
        WFSA_count=0
        for n, WFSA in enumerate(WFSA_delete):
            if not(WFSA.item()):
                new_weights[WFSA_count,:,:,:] = old_weights[n,:,:new_weights.shape[2],:]
                new_biases[WFSA_count,:,:,:] = old_biases[n,:,:new_biases.shape[2],:]
                new_epsilon[WFSA_count,:] = old_epsilon[n,:new_epsilon.shape[1]]
                WFSA_count+=1
        
        new_weights = new_weights.reshape(-1,new_weights.shape[-1])
        new_biases = new_biases.reshape(-1,new_biases.shape[-1])
        
        new_model.sopa.diags=Parameter(new_weights)
        new_model.sopa.bias=Parameter(new_biases)
        new_model.sopa.epsilon=Parameter(new_epsilon)
    
        return new_model, new_pattern_specs
    else:
        return None, None

def get_regularization_groups(model):
    n_weights_params = torch.ones_like(model.diags.view(model.total_num_patterns,-1,model.input_dim)).to(device)
    n_bias_params = torch.ones_like(model.bias.view(model.total_num_patterns,model.num_diags * model.max_pattern_length)).to(device)
    n_weights_params=n_weights_params.sum(dim=1).sum(dim=1)
    n_bias_params = n_bias_params.sum(dim=1)

    reshaped_weights = model.diags.view(model.total_num_patterns,-1,model.input_dim)
    reshaped_bias = model.bias.view(model.total_num_patterns,model.num_diags * model.max_pattern_length)
    l2_norm = reshaped_weights.norm(2, dim=1).norm(2, dim=1)/n_weights_params + reshaped_bias.norm(2, dim=1)/n_bias_params
    return l2_norm



def evaluate_decomp_deepsup_AUPRC(model, data_gen):
    with torch.no_grad():
        model.eval()
        #data_val_loss = []
        data_true = []
        data_pred = []
        for each_batch in range(data_gen.steps):
            data = next(data_gen)
            ts = data['ts']
            data = data['data']
            
            x = torch.tensor(data[0][0], dtype=torch.float32).to(device)
            mask = torch.tensor(data[0][1], dtype=torch.float32).unsqueeze(-1).to(device)
            y = torch.tensor(data[1], dtype=torch.float32).to(device)
            
            output = model(x.transpose(1,2),mask)
            output = output.unsqueeze(2)
            masked_output = output * mask
            
            for m, t, p in zip(mask.cpu().numpy().flatten(),
                               y.cpu().numpy().flatten(),
                               output.cpu().detach().numpy().flatten()):
                if np.equal(m, 1):
                    data_true.append(t)
                    data_pred.append(p)
        data_pred = np.array(data_pred)
        data_pred = np.stack([1 - data_pred, data_pred], axis=1)
        ret = metrics.print_metrics_binary(data_true, data_pred,0)
    return ret['auprc']

def train_reg_str(train_data_gen, dev_data_gen, model, num_epochs, learning_rate, run_scheduler, gpu, clip, patience, reg_strength, logging_path):
    from .my_utils import remove_old,to_file,enable_gradient_clipping
    """ Train a model on all the given docs """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    enable_gradient_clipping(model, clip)

    if run_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.1, 10, True)

    best_dev_loss = 100000000
    best_dev_loss_index = -1
    best_dev_AUPRC = 0
    
    unchanged = 0
    reduced_model_path = ""
    learned_pattern_specs = ""
    stop = False
    for each_chunk in range(num_epochs):
        cur_batch_loss = []
        model.train()
        for each_batch in range(train_data_gen.steps):
            optimizer.zero_grad()
            
            batch_data = next(train_data_gen)
            batch_ts = batch_data['ts']
            batch_data = batch_data['data']

            batch_x = torch.tensor(batch_data[0][0], dtype=torch.float32).to(device)
            batch_mask = torch.tensor(batch_data[0][1], dtype=torch.float32).unsqueeze(-1).to(device)
            batch_y = torch.tensor(batch_data[1], dtype=torch.float32).to(device)
            cur_output = model(batch_x.transpose(1,2),batch_mask)
            masked_output = cur_output.unsqueeze(2) * batch_mask         
            loss = batch_y * torch.log(masked_output + 1e-7) + (1 - batch_y) * torch.log(1 - masked_output + 1e-7)
            loss = torch.sum(loss, dim=1) / torch.sum(batch_mask, dim=1)
            loss = torch.neg(torch.sum(loss))

            
            #Regularization 
            regularization_term= get_regularization_groups(model.sopa)
            regularization_term = torch.sum(regularization_term)
            reg_loss = reg_strength * regularization_term
            train_loss = loss + reg_loss
        
            
            train_loss.backward()
            optimizer.step()
            
        model.eval()
        dev_AUPRC = evaluate_decomp_deepsup_AUPRC(model,dev_data_gen)
        print("dev_AUPRC ", dev_AUPRC)
            
        if best_dev_AUPRC < dev_AUPRC:
            unchanged = 0
            best_dev_AUPRC=dev_AUPRC
        else:
            unchanged += 1
        if unchanged >= patience:
            stop = True

        epoch_string = "\n"
        epoch_string += "-" * 110 + "\n"
        epoch_string += "| Epoch={} | reg_strength={} | train_loss={:.6f} | valid_acc={:.6f} | regularized_loss={:.6f} |".format(
            each_chunk,
            reg_strength,
            train_loss.item(),
            dev_AUPRC,
            reg_loss.item()
            )
        
        #new_model_valid_err = -1.0
        new_model, new_pattern_specs = extract_learned_structure(model)
        if new_model is not None:
            if new_pattern_specs != model.sopa.pattern_specs:
                new_model_dev_AUPRC = evaluate_decomp_deepsup_AUPRC(new_model, dev_data_gen)
            else:
                new_model_dev_AUPRC = dev_AUPRC
            epoch_string += " extracted_structure valid_err={:.6f} |".format(new_model_dev_AUPRC)
        print(epoch_string)
        
        if unchanged == 0:
            if reduced_model_path != "":
                remove_old(reduced_model_path)
         
            learned_pattern_specs, reduced_model_path = to_file(new_model,new_pattern_specs,logging_path)
            


        if stop:
            break
    return dev_AUPRC, learned_pattern_specs, reduced_model_path

def train_no_reg(train_data_gen, dev_data_gen, model, num_epochs, learning_rate, run_scheduler, gpu, clip, patience, logging_path):
    from .my_utils import remove_old,to_file,enable_gradient_clipping
    """ Train a model on all the given docs """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    enable_gradient_clipping(model, clip)

    if run_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.1, 10, True)

    best_dev_loss = 100000000
    best_dev_loss_index = -1
    best_dev_AUPRC = 0
    
    unchanged = 0
    reduced_model_path = ""
    learned_pattern_specs = ""
    stop = False
    for each_chunk in range(num_epochs):
        cur_batch_loss = []
        model.train()
        for each_batch in range(train_data_gen.steps):
            optimizer.zero_grad()
            
            batch_data = next(train_data_gen)
            batch_ts = batch_data['ts']
            batch_data = batch_data['data']

            batch_x = torch.tensor(batch_data[0][0], dtype=torch.float32).to(device)
            batch_mask = torch.tensor(batch_data[0][1], dtype=torch.float32).unsqueeze(-1).to(device)
            batch_y = torch.tensor(batch_data[1], dtype=torch.float32).to(device)
            cur_output = model(batch_x.transpose(1,2),batch_mask)
            masked_output = cur_output.unsqueeze(2) * batch_mask         
            loss = batch_y * torch.log(masked_output + 1e-7) + (1 - batch_y) * torch.log(1 - masked_output + 1e-7)
            loss = torch.sum(loss, dim=1) / torch.sum(batch_mask, dim=1)
            loss = torch.neg(torch.sum(loss))

            
            train_loss = loss
        
            
            train_loss.backward()
            optimizer.step()
            
        model.eval()
        dev_AUPRC = evaluate_decomp_deepsup_AUPRC(model,dev_data_gen)
        print("dev_AUPRC ", dev_AUPRC)
            
        if best_dev_AUPRC < dev_AUPRC:
            unchanged = 0
            best_dev_AUPRC=dev_AUPRC
        else:
            unchanged += 1
        if unchanged >= patience:
            stop = True

        epoch_string = "\n"
        epoch_string += "-" * 110 + "\n"
        epoch_string += "| Epoch={} | train_loss={:.6f} | valid_acc={:.6f} |".format(
            each_chunk,
            train_loss.item(),
            dev_AUPRC,
            )
        
        print(epoch_string)
        
        if unchanged == 0:
            if reduced_model_path != "":
                remove_old(reduced_model_path)
         
            learned_pattern_specs, reduced_model_path = to_file(model,model.sopa.pattern_specs,logging_path)
            


        if stop:
            break
    return dev_AUPRC, learned_pattern_specs, reduced_model_path