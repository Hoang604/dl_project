import math
import torch
import torch.nn as nn

class ViTEncoder(nn.Module):
    def __init__(self, x_dim, hid_dim, z_dim, patch_size=(7, 2), num_heads=4, num_layers=4, dropout=0.1):
        super(ViTEncoder, self).__init__()
        # x_dim is [channels, time, freq]
        self.input_features = [x_dim[1], x_dim[2]]
        
        self.patch_size = patch_size
        self.hid_dim = hid_dim
        self.z_dim = z_dim
        
        # Calculate number of patches
        time_len = x_dim[1]
        freq_len = x_dim[2]
        
        # Ensure padding matches patch size if not divisible
        pad_t = (patch_size[0] - (time_len % patch_size[0])) % patch_size[0]
        pad_f = (patch_size[1] - (freq_len % patch_size[1])) % patch_size[1]
        
        self.pad = nn.ZeroPad2d((0, pad_f, 0, pad_t)) # pad right and bottom
        
        padded_time = time_len + pad_t
        padded_freq = freq_len + pad_f
        
        num_patches = (padded_time // patch_size[0]) * (padded_freq // patch_size[1])
        
        # Patch projection layer
        self.patch_conv = nn.Conv2d(
            in_channels=1,
            out_channels=hid_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Learnable tokens & positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hid_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, hid_dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hid_dim,
            nhead=num_heads,
            dim_feedforward=hid_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.layernorm = nn.LayerNorm(hid_dim)
        
        # MLP Head to output dimension z_dim
        self.head = nn.Linear(hid_dim, z_dim)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        # Input shape: (batch_size, 1, time, freq)
        x = self.pad(x)
        
        # Patchify: (batch_size, hid_dim, patches_t, patches_f)
        x = self.patch_conv(x)
        
        # Flatten and transpose: (batch_size, num_patches, hid_dim)
        x = x.flatten(2).transpose(1, 2)
        
        # Prepend class token
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embeddings
        x = x + self.pos_embedding
        
        # Transformer forward pass
        x = self.transformer(x)
        
        # Extract representation of CLS token
        x = self.layernorm(x[:, 0])
        
        # Project to target dimension
        x = self.head(x)
        
        return x
