import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from domainbed.networks import AdversarialPertubation, AdversarialPertubation2
from torchvision import transforms
import torchvision

import copy

from domainbed import networks


ALGORITHMS = [
    'ERM',
    'FedAvg',
    'FedIIR',
    'FedSVD'
]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)

# FedAvg
class FedAvg(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams, args = None):
        super(FedAvg, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.device = args.device

    def create_client(self,):
        self.featurizer_client = copy.deepcopy(self.featurizer)
        self.classifier_client = copy.deepcopy(self.classifier)
        self.optimizer_client = torch.optim.SGD(
            list(self.featurizer_client.parameters()) + list(self.classifier_client.parameters()),
            lr=self.hparams["lr"],
            momentum=0.9,
            weight_decay=self.hparams['weight_decay']
        )
        
    def aggregation_client(self, model_client):

        def aggregation(weights):
    
            weights_avg = copy.deepcopy(weights[0])
            for k in weights_avg.keys():
                for i in range(1, num_client):
                    weights_avg[k] += weights[i][k]
                weights_avg[k] = torch.div(weights_avg[k], num_client)
            
            return weights_avg        
        
        num_model = len(model_client[0])  # number of model
        num_client = len(model_client)      # the number of client
        weights_avg = []
        for i in range(num_model):
            weights = []
            for _, total_weights in enumerate(model_client):
                weights.append(total_weights[i])
            weights_avg.append(aggregation(weights))
        
        self.featurizer.load_state_dict(weights_avg[0])
        self.classifier.load_state_dict(weights_avg[1])


    def update(self, sampled_clients, steps):

        model_client = []
        for _, client_data in sampled_clients:
            client_model_dict = {}
            self.create_client()
            
            for step in range(steps):
                for x, y in client_data:
                    x, y = x.to(self.device), y.to(self.device)

                    self.optimizer_client.zero_grad()
                    feature = self.featurizer_client(x)
                    logits = self.classifier_client(feature)
                    loss = F.cross_entropy(logits, y)
                    loss.backward()
                    self.optimizer_client.step()
            
            client_model_dict['F'] = self.featurizer_client.state_dict()
            client_model_dict['C'] = self.classifier_client.state_dict()
            model_client.append([client_model_dict['F'], client_model_dict['C']])
        self.aggregation_client(model_client)

        return {'loss': loss.item()}
    
    def predict(self, x):
        return self.classifier(self.featurizer(x))


# FedSVD
class FedSVD(FedAvg):
    def __init__(self, input_shape, num_classes, num_domains, hparams, args = None):
        super(FedSVD, self).__init__(input_shape, num_classes, num_domains, hparams, args)
        # self.args = args
        self.perturb_type = hparams['perturb_type']
        
        # self.global_epoch = 0

    def get_local_perturbed_output(self, feature, logits):
        adv_perturb = AdversarialPertubation(self.classifier_client, self.device, self.hparams)
        if self.perturb_type == 'weight':
            logits_perturb = adv_perturb.weight_perturb_predict(feature, logits)
        # else:
        #     raise NotImplementedError
        elif self.perturb_type == 'singular':
            logits_perturb = adv_perturb.singular_perturb_predict(feature, logits)
        return logits_perturb
    
    def get_global_perturbed_output(self, x, logits):
        with torch.no_grad():
            feature = self.featurizer(x)

        adv_perturb = AdversarialPertubation(self.classifier, self.device, self.hparams)
        if self.perturb_type == 'weight':
            logits_perturb = adv_perturb.weight_perturb_predict(feature, logits)
        # else:
        #     raise NotImplementedError
        elif self.perturb_type == 'singular':
            logits_perturb = adv_perturb.singular_perturb_predict(feature, logits)
        return logits_perturb

    def update(self, sampled_clients, steps):

        model_client = []
        for _, client_data in sampled_clients:
            client_model_dict = {}
            self.create_client()
            
            for step in range(steps):
                for x, y in client_data:
                    x, y = x.to(self.device), y.to(self.device)

                    self.optimizer_client.zero_grad()

                    # 1. cross-entropy loss
                    feature = self.featurizer_client(x)
                    logits = self.classifier_client(feature)
                    loss = F.cross_entropy(logits, y)

                    # 2. local smoothness
                    if self.hparams['local_smooth'] > 0:
                        logits_local_perturb = self.get_local_perturbed_output(feature.detach(), logits.detach())
                        loss_local_smooth = F.kl_div(F.log_softmax(logits, dim=1), F.softmax(logits_local_perturb, dim=1), reduction='batchmean')
                        loss += self.hparams['local_smooth'] * loss_local_smooth

                    # 3. global smoothness
                    if self.hparams['global_smooth'] > 0:
                        logits_global_perturb = self.get_global_perturbed_output(x, logits.detach())
                        loss_global_smooth = F.kl_div(F.log_softmax(logits, dim=1), F.softmax(logits_global_perturb, dim=1), reduction='batchmean')
                        loss += self.hparams['global_smooth'] * loss_global_smooth

                    loss.backward()
                    self.optimizer_client.step()
            
            client_model_dict['F'] = self.featurizer_client.state_dict()
            client_model_dict['C'] = self.classifier_client.state_dict()
            model_client.append([client_model_dict['F'], client_model_dict['C']])
        self.aggregation_client(model_client)

        return {'loss': loss.item()}


# FedSVD2: more layers including conv2d
class FedSVD2(FedAvg):
    def __init__(self, input_shape, num_classes, num_domains, hparams, args = None):
        super(FedSVD2, self).__init__(input_shape, num_classes, num_domains, hparams, args)
        # self.args = args
        self.perturb_type = hparams['perturb_type']
        
    def get_local_perturbed_output(self, x, logits):
        adv_perturb = AdversarialPertubation2(copy.deepcopy(self.featurizer_client), 
                                              copy.deepcopy(self.classifier_client), 
                                              self.device, self.hparams)
        if self.perturb_type == 'weight':
            logits_perturb = adv_perturb.weight_perturb_predict(x, logits)
        else:
            raise NotImplementedError
        # elif self.perturb_type == 'singular':
        #     logits_perturb = adv_perturb.singular_perturb_predict(feature, logits)
        return logits_perturb
    
    def get_global_perturbed_output(self, x, logits):
        adv_perturb = AdversarialPertubation2(copy.deepcopy(self.featurizer), 
                                              copy.deepcopy(self.classifier), 
                                              self.device, self.hparams)
        if self.perturb_type == 'weight':
            logits_perturb = adv_perturb.weight_perturb_predict(x, logits)
        else:
            raise NotImplementedError
        # elif self.perturb_type == 'singular':
        #     logits_perturb = adv_perturb.singular_perturb_predict(feature, logits)
        return logits_perturb

    def update(self, sampled_clients, steps):

        model_client = []
        for _, client_data in sampled_clients:
            client_model_dict = {}
            self.create_client()
            
            for step in range(steps):
                for x, y in client_data:
                    x, y = x.to(self.device), y.to(self.device)

                    self.optimizer_client.zero_grad()

                    # 1. cross-entropy loss
                    feature = self.featurizer_client(x)
                    logits = self.classifier_client(feature)
                    loss = F.cross_entropy(logits, y)

                    # 2. local smoothness
                    logits_local_perturb = self.get_local_perturbed_output(x, logits.detach())
                    loss_local_smooth = F.kl_div(F.log_softmax(logits, dim=1), F.softmax(logits_local_perturb, dim=1), reduction='batchmean')
                    loss += self.hparams['local_smooth'] * loss_local_smooth

                    # 3. global smoothness
                    logits_global_perturb = self.get_global_perturbed_output(x, logits.detach())
                    loss_global_smooth = F.kl_div(F.log_softmax(logits, dim=1), F.softmax(logits_global_perturb, dim=1), reduction='batchmean')
                    loss += self.hparams['global_smooth'] * loss_global_smooth

                    loss.backward()
                    self.optimizer_client.step()
            
            client_model_dict['F'] = self.featurizer_client.state_dict()
            client_model_dict['C'] = self.classifier_client.state_dict()
            model_client.append([client_model_dict['F'], client_model_dict['C']])
        self.aggregation_client(model_client)

        return {'loss': loss.item()}


# FedIIR
class FedIIR(FedAvg):
    def __init__(self, input_shape, num_classes, num_domains, hparams, args = None):
        super(FedIIR, self).__init__(input_shape, num_classes, num_domains, hparams, args)

        self.global_epoch = 0
        params = list(self.classifier.parameters())
        self.grad_mean = tuple(torch.zeros_like(p).to(self.device) for p in params)

    def mean_grad(self, sampled_clients):
        
        total_batch = 0
        grad_sum = tuple(torch.zeros_like(g).to(self.device) for g in self.grad_mean)
        for _, client_data in sampled_clients:
            
            for x, y in client_data:
                x, y = x.to(self.device), y.to(self.device)
                feature = self.featurizer(x)
                logits = self.classifier(feature)
                loss = F.cross_entropy(logits, y)
                grad_batch = autograd.grad(loss, self.classifier.parameters(), create_graph=False)

                grad_sum = tuple(g1 + g2 for g1, g2 in zip(grad_sum, grad_batch))
                total_batch += 1
        
        grad_mean_new = tuple(grad / total_batch for grad in grad_sum)
        return tuple(self.hparams['ema'] * g1 + (1 - self.hparams['ema']) * g2 
                     for g1, g2 in zip(self.grad_mean, grad_mean_new))
            

    def update(self, sampled_clients, steps):
        
        penalty_weight = self.hparams['penalty']
        self.grad_mean = self.mean_grad(sampled_clients)
        model_client = []
        for _, client_data in sampled_clients:
            client_model_dict = {}
            self.create_client()

            for step in range(steps):
                for x, y in client_data:
                    x, y = x.to(self.device), y.to(self.device)

                    self.optimizer_client.zero_grad()

                    feature = self.featurizer_client(x)
                    logits = self.classifier_client(feature)
                    
                    loss_erm = F.cross_entropy(logits, y)
                    grad_client = autograd.grad(loss_erm, self.classifier_client.parameters(), create_graph=True)
                    # compute trace penalty
                    penalty_value = 0
                    for g_client, g_mean in zip(grad_client,self.grad_mean):
                        penalty_value += (g_client - g_mean).pow(2).sum()
                    loss = loss_erm + penalty_weight * penalty_value

                    loss.backward()
                    self.optimizer_client.step()
            
            client_model_dict['F'] = self.featurizer_client.state_dict()
            client_model_dict['C'] = self.classifier_client.state_dict()
            model_client.append([client_model_dict['F'], client_model_dict['C']])
        self.aggregation_client(model_client)

        self.global_epoch += 1

        return {'loss': loss_erm.item(), 'penalty': penalty_value.item()}
