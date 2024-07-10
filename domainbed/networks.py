import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torchvision.models
from torchvision.models.resnet import BasicBlock, Bottleneck

def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        if hparams['resnet18']:
            self.network = torchvision.models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
            self.n_outputs = 2048

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.network = nn.Sequential()
        self.network.fc = Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        x = self.network(x)
        return x


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (224, 224):
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        network = torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, out_features))
    else:
        network = torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features))

    def initialize_paras(network):
        for layer in network:
            if isinstance(layer,torch.nn.Linear):
                init.kaiming_normal_(layer.weight)
    initialize_paras(network)

    return network


class AdversarialPertubation:
    def __init__(self, classifier, device, hparams):
        # self.featurizer = featurizer
        self.classifier = classifier

        # self.perturb_init_scale = 0.1
        # self.perturb_grad_scale = 0.01
        self.device = device
        self.hparams = hparams

    def weight_perturb_predict(self, feature, logits):
        # add perturbation to weight, and then forward
        #----- step 1: generate random perturbation -----#
        pertub_layers = []
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):

                weight = layer.weight.data

                # generate random perturbation
                if self.hparams['perturb_dist'] == 'uniform':
                    delta = torch.rand(weight.shape).sub(0.5).to(self.device)
                elif self.hparams['perturb_dist'] == 'normal':
                    delta = torch.randn(weight.shape).to(self.device)
                # normalize to unit ball
                delta = delta.div(torch.norm(delta, p=2, dim=1, keepdim=True) + 1e-8)
                # require grad
                delta.requires_grad = True
                
                # not perturb bias
                bias = layer.bias.data

                pertub_layers.append((weight, delta, bias))
            else:
                pertub_layers.append(layer)

        #----- step 2: forward with perturbation -----#
        # z = self.featurizer(x)
        z = feature
        for layer in pertub_layers:
            if isinstance(layer, tuple):
                weight, delta, bias = layer
                z = F.linear(z, weight + self.hparams['perturb_init_scale'] * delta, bias)
            else:
                z = layer(z)
        logits_perturb = z

        # calculate KL div loss
        loss_kl = F.kl_div(F.log_softmax(logits_perturb, dim=1), F.softmax(logits, dim=1), reduction='batchmean')
        loss_kl.backward()
        
        #----- step 3: forward with new perturbation -----#
        z = feature
        for layer in pertub_layers:
            if isinstance(layer, tuple):
                weight, delta, bias = layer
                grad = delta.grad
                grad = grad.div(torch.norm(grad, p=2, dim=1, keepdim=True) + 1e-8)
                z = F.linear(z, weight + self.hparams['perturb_grad_scale'] * grad, bias)
                # import pdb; pdb.set_trace()
            else:
                z = layer(z)
        logits_perturb = z
        # loss_kl = - F.kl_div(F.log_softmax(logits_perturb, dim=1), F.softmax(logits, dim=1), reduction='batchmean')
        return logits_perturb.detach()

    def singular_perturb_predict(self, feature, logits):
        # add perturbation to singular value of weight, and then forward
        #----- step 1: generate random perturbation -----#
        pertub_layers = []
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):

                weight = layer.weight.data

                # svd of weight
                U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

                # generate random perturbation
                if self.hparams['perturb_dist'] == 'uniform':
                    delta = torch.rand(S.shape).sub(0.5).to(self.device)
                elif self.hparams['perturb_dist'] == 'normal':
                    delta = torch.randn(S.shape).to(self.device)
                # normalize to unit ball
                delta = delta.div(torch.norm(delta, p=2) + 1e-8)
                # require grad
                delta.requires_grad = True
                
                # not perturb bias
                bias = layer.bias.data

                pertub_layers.append((U, S, Vh, delta, bias))
            else:
                pertub_layers.append(layer)

        #----- step 2: forward with perturbation -----#
        # z = self.featurizer(x)
        z = feature
        for layer in pertub_layers:
            if isinstance(layer, tuple):
                U, S, Vh, delta, bias = layer
                S_ = F.relu(S + self.hparams['perturb_init_scale'] * delta)
                weight = U @ torch.diag(S_) @ Vh
                z = F.linear(z, weight, bias)
            else:
                z = layer(z)
        logits_perturb = z

        # calculate KL div loss
        loss_kl = F.kl_div(F.log_softmax(logits_perturb, dim=1), F.softmax(logits, dim=1), reduction='batchmean')
        loss_kl.backward()
        
        #----- step 3: forward with new perturbation -----#
        z = feature
        for layer in pertub_layers:
            if isinstance(layer, tuple):
                U, S, Vh, delta, bias = layer
                grad = delta.grad
                grad = grad.div(torch.norm(grad, p=2) + 1e-8)
                S_ = F.relu(S + self.hparams['perturb_grad_scale'] * grad)
                weight = U @ torch.diag(S_) @ Vh
                z = F.linear(z, weight, bias)
                # import pdb; pdb.set_trace()
            else:
                z = layer(z)
        logits_perturb = z
        # loss_kl = - F.kl_div(F.log_softmax(logits_perturb, dim=1), F.softmax(logits, dim=1), reduction='batchmean')
        return logits_perturb.detach()


class AdversarialPertubation2:
    def __init__(self, featurizer, classifier, device, hparams):
        self.featurizer = featurizer
        self.classifier = classifier
        self.device = device
        self.hparams = hparams

    def weight_perturb_predict(self, x, logits):
        # add perturbation to weight, and then forward
        if 'layer4' in self.hparams['perturb_layers']:
            if isinstance(self.featurizer, ResNet):
                if isinstance(self.featurizer.network.layer4[-1], BasicBlock):
                    self.featurizer.network.layer4[-1].conv1 = WeightPerturbConv2d(self.featurizer.network.layer4[-1].conv1, 
                                                                                   self.hparams['perturb_dist'],
                                                                                   self.hparams['perturb_init_scale'], 
                                                                                   self.hparams['perturb_grad_scale'],
                                                                                   device=self.device)
                    self.featurizer.network.layer4[-1].conv2 = WeightPerturbConv2d(self.featurizer.network.layer4[-1].conv2, 
                                                                                   self.hparams['perturb_dist'],
                                                                                   self.hparams['perturb_init_scale'], 
                                                                                   self.hparams['perturb_grad_scale'],
                                                                                   device=self.device)
                elif isinstance(self.featurizer.network.layer4[-1], Bottleneck):
                    self.featurizer.network.layer4[-1].conv1 = WeightPerturbConv2d(self.featurizer.network.layer4[-1].conv1, 
                                                                                   self.hparams['perturb_dist'],
                                                                                   self.hparams['perturb_init_scale'], 
                                                                                   self.hparams['perturb_grad_scale'],
                                                                                   device=self.device)
                    self.featurizer.network.layer4[-1].conv2 = WeightPerturbConv2d(self.featurizer.network.layer4[-1].conv2, 
                                                                                   self.hparams['perturb_dist'],
                                                                                   self.hparams['perturb_init_scale'], 
                                                                                   self.hparams['perturb_grad_scale'],
                                                                                   device=self.device)
                    self.featurizer.network.layer4[-1].conv3 = WeightPerturbConv2d(self.featurizer.network.layer4[-1].conv3, 
                                                                                   self.hparams['perturb_dist'],
                                                                                   self.hparams['perturb_init_scale'], 
                                                                                   self.hparams['perturb_grad_scale'],
                                                                                   device=self.device)
            elif isinstance(self.featurizer, MNIST_CNN):
                self.featurizer.conv4 = WeightPerturbConv2d(self.featurizer.conv4, 
                                                            self.hparams['perturb_dist'],
                                                            self.hparams['perturb_init_scale'], 
                                                            self.hparams['perturb_grad_scale'],
                                                            device=self.device)
        if 'classifier' in self.hparams['perturb_layers']:
            for idx, layer in enumerate(self.classifier):
                if isinstance(layer, nn.Linear):
                    self.classifier[idx] = WeightPerturbLinear(layer, self.hparams['perturb_dist'], 
                                                            self.hparams['perturb_init_scale'], 
                                                            self.hparams['perturb_grad_scale'],
                                                            device=self.device)
        feature = self.featurizer(x)
        logits_perturb = self.classifier(feature)

        # calculate KL div loss and backward
        loss_kl = F.kl_div(F.log_softmax(logits_perturb, dim=1), F.softmax(logits, dim=1), reduction='batchmean')
        loss_kl.backward()
        
        # update weight and forward again
        for idx, layer in enumerate(self.classifier):
            if isinstance(layer, WeightPerturbLinear):
                layer.update_weight()
        for module in self.featurizer.modules():
            if isinstance(module, WeightPerturbConv2d):
                module.update_weight()
        feature = self.featurizer(x)
        logits_perturb = self.classifier(feature)

        return logits_perturb.detach()

    def singular_perturb_predict(self, feature, logits):
        # add perturbation to singular value of weight, and then forward
        #----- step 1: generate random perturbation -----#
        pertub_layers = []
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):

                weight = layer.weight.data

                # svd of weight
                U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

                # generate random perturbation
                if self.hparams['perturb_dist'] == 'uniform':
                    delta = torch.rand(S.shape).sub(0.5).to(self.device)
                elif self.hparams['perturb_dist'] == 'normal':
                    delta = torch.randn(S.shape).to(self.device)
                # normalize to unit ball
                delta = delta.div(torch.norm(delta, p=2))
                # require grad
                delta.requires_grad = True
                
                # not perturb bias
                bias = layer.bias.data

                pertub_layers.append((U, S, Vh, delta, bias))
            else:
                pertub_layers.append(layer)

        #----- step 2: forward with perturbation -----#
        # z = self.featurizer(x)
        z = feature
        for layer in pertub_layers:
            if isinstance(layer, tuple):
                U, S, Vh, delta, bias = layer
                weight = U @ torch.diag(S + self.hparams['perturb_init_scale'] * delta) @ Vh
                z = F.linear(z, weight, bias)
            else:
                z = layer(z)
        logits_perturb = z

        # calculate KL div loss
        loss_kl = F.kl_div(F.log_softmax(logits_perturb, dim=1), F.softmax(logits, dim=1), reduction='batchmean')
        loss_kl.backward()
        
        #----- step 3: forward with new perturbation -----#
        z = feature
        for layer in pertub_layers:
            if isinstance(layer, tuple):
                U, S, Vh, delta, bias = layer
                grad = delta.grad
                grad = grad.div(torch.norm(grad, p=2) + 1e-8)
                weight = U @ torch.diag(S + self.hparams['perturb_grad_scale'] * grad) @ Vh
                z = F.linear(z, weight, bias)
                # import pdb; pdb.set_trace()
            else:
                z = layer(z)
        logits_perturb = z
        # loss_kl = - F.kl_div(F.log_softmax(logits_perturb, dim=1), F.softmax(logits, dim=1), reduction='batchmean')
        return logits_perturb.detach()


class WeightPerturbLinear(nn.Module):
    # define a custom linear layer with weight and bias
    def __init__(self, linear:nn.Linear, perturb_dist, perturb_init_scale, perturb_grad_scale, device):
        super(WeightPerturbLinear, self).__init__()
        self.device = device
        self.perturb_dist = perturb_dist
        self.perturb_init_scale = perturb_init_scale
        self.perturb_grad_scale = perturb_grad_scale
        self.init_weight(linear)

    def init_weight(self, linear):
        weight = linear.weight.data

        # generate random perturbation
        if self.perturb_dist == 'uniform':
            delta = torch.rand(weight.shape).sub(0.5).to(self.device)
        elif self.perturb_dist == 'normal':
            delta = torch.randn(weight.shape).to(self.device)
        # normalize to unit ball
        delta = delta.div(torch.norm(delta, p=2, dim=1, keepdim=True) + 1e-8)
        # require grad
        delta.requires_grad = True

        self.delta = delta
        self.weight = weight + self.perturb_init_scale * delta
        self.bias = linear.bias.data
    
    def update_weight(self):
        grad = self.delta.grad
        grad = grad.div(torch.norm(grad, p=2, dim=1, keepdim=True) + 1e-8)
        self.weight = self.weight + self.perturb_grad_scale * grad

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)


class WeightPerturbConv2d(nn.Module):
    # define a custom linear layer with weight and bias
    def __init__(self, conv_layer:nn.Conv2d, perturb_dist, perturb_init_scale, perturb_grad_scale, device):
        super(WeightPerturbConv2d, self).__init__()
        self.device = device
        self.perturb_dist = perturb_dist
        self.perturb_init_scale = perturb_init_scale
        self.perturb_grad_scale = perturb_grad_scale
        self.init_weight(conv_layer)

    def init_weight(self, conv_layer):
        weight = conv_layer.weight.data

        # generate random perturbation
        if self.perturb_dist == 'uniform':
            delta = torch.rand(weight.shape).sub(0.5).to(self.device)
        elif self.perturb_dist == 'normal':
            delta = torch.randn(weight.shape).to(self.device)
        # normalize to unit ball
        delta = delta.div(torch.norm(delta, p=2, dim=(1,2,3), keepdim=True) + 1e-8)
        # require grad
        delta.requires_grad = True

        self.delta = delta
        self.weight = weight + self.perturb_init_scale * delta
        self.bias = conv_layer.bias

        self.stride = conv_layer.stride
        self.padding = conv_layer.padding
        self.dilation = conv_layer.dilation
        self.groups = conv_layer.groups
    
    def update_weight(self):
        grad = self.delta.grad
        grad = grad.div(torch.norm(grad, p=2, dim=(1,2,3), keepdim=True) + 1e-8)
        self.weight = self.weight + self.perturb_grad_scale * grad

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
