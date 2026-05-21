import sys
sys.path.append('/home/hoang/python/dl_project/KWSFSL')

import torch
from models.encoder.DSCNN import DSCNNL_LAYERNORM
from models.encoder.ViT import ViTEncoder
from ptflops import get_model_complexity_info

def profile_models():
    x_dim = [1, 49, 10]
    
    print("=" * 60)
    print("PROFILING CURRENT MODEL: DSCNNL_LAYERNORM")
    print("=" * 60)
    dscnn_model = DSCNNL_LAYERNORM(x_dim)
    
    macs_dscnn, params_dscnn = get_model_complexity_info(
        dscnn_model, (1, 49, 10), as_strings=True, print_per_layer_stat=False, verbose=False
    )
    print(f"DSCNNL MACs (operations): {macs_dscnn}")
    print(f"DSCNNL Parameters:       {params_dscnn}")
    
    # Calculate exact parameter count and size in MB
    num_params_dscnn = sum(p.numel() for p in dscnn_model.parameters())
    size_mb_dscnn = (num_params_dscnn * 4) / (1024 * 1024)
    print(f"DSCNNL Exact Param Count: {num_params_dscnn:,}")
    print(f"DSCNNL Size on Disk:      {size_mb_dscnn:.3f} MB")
    
    print("\n" + "=" * 60)
    print("PROFILING NEW LAPTOP MODEL: ViTEncoder (Large config)")
    print("=" * 60)
    vit_model = ViTEncoder(x_dim=x_dim, hid_dim=256, z_dim=128)
    
    macs_vit, params_vit = get_model_complexity_info(
        vit_model, (1, 49, 10), as_strings=True, print_per_layer_stat=False, verbose=False
    )
    print(f"ViT MACs (operations):    {macs_vit}")
    print(f"ViT Parameters:          {params_vit}")
    
    # Calculate exact parameter count and size in MB
    num_params_vit = sum(p.numel() for p in vit_model.parameters())
    size_mb_vit = (num_params_vit * 4) / (1024 * 1024)
    print(f"ViT Exact Param Count:    {num_params_vit:,}")
    print(f"ViT Size on Disk:         {size_mb_vit:.3f} MB")
    
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print(f"Parameter Ratio (ViT / DSCNNL): {num_params_vit / num_params_dscnn:.2f}x larger")
    
if __name__ == "__main__":
    profile_models()
