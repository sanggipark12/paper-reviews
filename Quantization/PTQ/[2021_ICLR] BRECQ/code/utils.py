import torch
import torch.nn as nn
from modules import UniformQuantizer, AdaRoundQuantizer, QuantModule

def bn_folding(conv, bn):
    # bn params
    mu = bn.running_mean
    var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps

    # conv params
    w = conv.weight
    b = conv.bias if conv.bias is not None else torch.zeros_like(mu)

    denominator = torch.sqrt(var + eps)

    w_new = w * (gamma / denominator).view(-1, 1, 1, 1)
    b_new = (b - mu) * (gamma / denominator) + beta

    conv.weight.data.copy_(w_new)

    if conv.bias is not None:
        conv.bias.data.copy_(b_new)
    else:
        conv.bias = nn.Parameter(b_new)

    return nn.Identity()


def fuse_resnet_module(model):
    model.eval()

    if hasattr(model, 'bn1') and not isinstance(model.bn1, nn.Identity):
        model.bn1 = bn_folding(model.conv1, model.bn1)


    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(model, layer_name)
        for i, block in enumerate(layer):
            print(f'Fusing {layer_name}_{i}')

            block.bn1 = bn_folding(block.conv1, block.bn1)
            block.bn2 = bn_folding(block.conv2, block.bn2)

            if block.downsample is not None:
                ## 0이 conv 1이 bn
                block.downsample[1] = bn_folding(block.downsample[0], block.downsample[1])

    print(" Folding completed")
    return model


def replace_to_quant_module(model):
    """
    모델을 재귀적으로 탐색하며 nn.Conv2d를 QuantModule로 교체
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):

            # UniformQuantizer 먼저 만들고 -> AdaRoundQuantizer로 감싸기
            uq = UniformQuantizer(n_bits=4, symmetric=True, channel_wise=True, is_weight=True)

            # 초기화를 위해 가중치 한번 넣어줌 (init_quantization_params)
            uq.init_quantization_params(module.weight)

            # AdaRound Quantizer 생성
            ada_quantizer = AdaRoundQuantizer(uq, module.weight)

            # Wrapper로 교체
            quant_module = QuantModule(module, ada_quantizer)
            setattr(model, name, quant_module)

        elif len(list(module.children())) > 0:
            # 자식 모듈이 더 있으면 재귀 호출
            replace_to_quant_module(module)
            
            