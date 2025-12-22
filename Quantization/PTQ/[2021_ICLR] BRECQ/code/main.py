import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from utils import fuse_resnet_module, replace_to_quant_module
from brecq import run_brecq

def main():
    # 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 모델 로드 및 전처리
    print("Loading ResNet18...")
    model = models.resnet18(pretrained=True)
        
    model.to(device)

    # BN Folding
    model = fuse_resnet_module(model)

    # QuantModule 교체
    replace_to_quant_module(model)
    model.to(device)

    # 데이터셋 준비 (CIFAR10)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # BRECQ 실행
    # Layer1의 첫 번째 블록 타겟팅
    target_block = model.layer1[0]
    print(f"\nStart Processing Block: {target_block}")

    run_brecq(model, dataloader, target_block, device)

    # 결과 검증
    print("\n=== Validation ===")
    q_module = target_block.conv1
    alpha = q_module.weight_quantizer.alpha
    
    print(f"Learned Alpha (First 5): {alpha.view(-1)[:5].detach().cpu().numpy()}")
    
    soft_mask = q_module.weight_quantizer.rectified_sigmoid()
    print(f"Soft Mask (Prob): {soft_mask.view(-1)[:5].detach().cpu().numpy()}")
    
    final_decision = (q_module.weight_quantizer.alpha >= 0).float()
    print(f"Final Decision:   {final_decision.view(-1)[:5].detach().cpu().numpy()}")

if __name__ == "__main__":
    main()