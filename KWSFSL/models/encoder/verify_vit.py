import sys
sys.path.append('/home/hoang/python/dl_project/KWSFSL')

import torch
from models.repr_model import get_encoder

def test_vit():
    print("Initializing ViT via get_encoder...")
    x_dim = [1, 49, 10]
    hid_dim = 256
    z_dim = 128
    
    # Test setting custom arguments as passed from command-line options
    model = get_encoder(
        'ViT', x_dim, hid_dim, z_dim,
        patch_size='4,1', num_heads=8, num_layers=6, dropout=0.2
    )
    print(model)
    
    # Synthetic batch of size 4
    x = torch.randn(4, 1, 49, 10)
    print("Input shape:", x.shape)
    
    out = model(x)
    print("Output shape:", out.shape)
    
    assert out.shape == (4, 128), f"Expected shape (4, 128), got {out.shape}"
    print("Verification SUCCESSful!")

if __name__ == "__main__":
    test_vit()
