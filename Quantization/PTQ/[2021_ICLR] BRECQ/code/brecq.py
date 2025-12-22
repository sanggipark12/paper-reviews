import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from modules import QuantModule

class DataSaver:
    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.grads = []

    def hook_fn(self, module, input, output):
        # .clone()을 추가하여 메모리상에서 완전히 분리
        self.inputs.append(input[0].detach().clone().cpu())
        self.outputs.append(output.detach().clone().cpu())

    def hook_backward(self, module, grad_input, grad_output):
        self.grads.append(grad_output[0].detach().clone().cpu())
        
        
def get_quantizers_from_block(block):
    """블록 내부의 모든 QuantModule에서 weight_quantizer를 추출"""
    quantizers = []
    for name, module in block.named_modules():
        if isinstance(module, QuantModule):
            quantizers.append(module.weight_quantizer)
    return quantizers

def block_reconstruction(block, cali_inputs, cali_outputs, cali_grads, quantizers, device, batch_size=32, iters=1000):
    """
    cali_inputs: List[Tensor] or Tensor
    cali_outputs: List[Tensor] or Tensor (FP32 출력)
    cali_grads: List[Tensor] or Tensor (Gradient)
    """

    # 입력 데이터가 리스트라면 하나로 합쳐줍니다. 
    if isinstance(cali_inputs, list):
        cali_inputs = torch.cat(cali_inputs, dim=0)
    if isinstance(cali_outputs, list):
        cali_outputs = torch.cat(cali_outputs, dim=0)
    if isinstance(cali_grads, list):
        cali_grads = torch.cat(cali_grads, dim=0)

    # 학습 파라미터 추출 및 adaround 설정
    params = []
    for q in quantizers:
        params.append(q.alpha)
        q.soft_targets = True

    optimizer = Adam(params, lr=1e-3)

    dataset = TensorDataset(cali_inputs.cpu(), cali_outputs.cpu(), cali_grads.cpu())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    beta_scheduler = np.linspace(20, 2, iters)

    for i in range(iters):
        beta = beta_scheduler[i]

        for x_batch, y_batch, g_batch in loader:
            # 여기서 데이터를 GPU로 이동하여 메모리 절약
            cur_input = x_batch.to(device) # Input
            cur_target = y_batch.to(device) # Output (Target)
            cur_grad = g_batch.to(device) # Fisher (Weight)

            optimizer.zero_grad()

            # Forward (Soft Quantization)
            out_quant = block(cur_input)

            # Fisher Loss 계산 
            # 수식: sum( (output_diff * grad)^2 )
            delta = out_quant - cur_target

            loss_rec = (delta * cur_grad).pow(2).sum()

            # 데이터 개수로 정규화 (안정성 위해)
            loss_rec = loss_rec / batch_size

            # Regularization Loss 
            loss_reg = 0

            for q in quantizers:
                soft_val = q.rectified_sigmoid()
                reg_term = 1.0 - (2 * soft_val - 1).abs().pow(beta)
                loss_reg += reg_term.sum()

            # Total Loss
            total_loss = loss_rec + 1e-4 * loss_reg

            total_loss.backward()
            optimizer.step()

            if i % 200 == 0:
                print(f"Iter {i}: Total {total_loss.item():.4f} (Rec {loss_rec.item():.4f})")
        
        break # 테스트만

    # 학습 종료 후 Hard Mode로 전환
    for q in quantizers:
        q.soft_targets = False
        
        
def run_brecq(model, dataloader, target_block, device):
    model.eval().to(device)
    criterion = nn.CrossEntropyLoss()
    saver = DataSaver()

    # Hook 등록
    # forward: 입력(input)과 정답(output) 수집
    h1 = target_block.register_forward_hook(saver.hook_fn)
    # backward: 중요도(gradient) 수집
    h2 = target_block.register_full_backward_hook(saver.hook_backward)

    print("Step 1: Collecting Data & Gradients...")


    for imgs, labels in dataloader:

        imgs, labels = imgs.to(device), labels.to(device)
        model.zero_grad()

        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()


    # Hook 제거 
    h1.remove()
    h2.remove()


    # block_reconstruction 내부에서 GPU로 옮길 것이므로 여기선 CPU로 둡니다
    cached_inputs = torch.cat(saver.inputs).cpu()
    cached_outputs = torch.cat(saver.outputs).cpu()
    cached_grads = torch.cat(saver.grads).cpu()


    print(f"  - Data Shape: {cached_inputs.shape}")
    print(f"  - Grad Shape: {cached_grads.shape}")

    # Quantizer 추출
    quantizers = get_quantizers_from_block(target_block)
    print(f"  - Found {len(quantizers)} quantizers in this block.")

    print("Step 2: Optimizing Block...")
    # 최적화 실행

    # block_reconstruction 인자: (block, inputs, outputs, grads, quantizers)
    # inputs를 list로 감싸서 전달 (함수 내부 호환성 위해)
    block_reconstruction(
        target_block,
        cached_inputs, cached_outputs, cached_grads,
        quantizers,
        device,
        iters=200 # 테스트니까 조금만
    )
    print("Done!")