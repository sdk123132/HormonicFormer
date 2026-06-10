"""
HormonicFormer v7r3 - CIFAR-10 适配版本
在 hormonic_v7r3_validated.py 基础上添加 CIFAR-10 支持
"""
import sys
sys.path.insert(0, r'C:\Users\MR\Desktop')

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from hormonic_v7r3_validated import HormonicFormerV7r3


class HormonicCIFAR10(nn.Module):
    """HormonicFormer for CIFAR-10 Classification"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config['model']['d_model']
        self.seq_len = 196  # 14x14 patches
        
        # Patch embedding: 2x2 patches from 32x32 image = 16x16 = 256 patches... 
        # Wait, 32/2=16, so 16x16=256 patches, but we want 196 (14x14)
        # Let's use 4x4 patches: 32/4=8, 8x8=64 patches... 
        # Actually let's use 2x2 with stride 2: 32/2=16, but we can do 14x14 by center crop
        
        # Simpler: 32x32 -> 4x4 patches = 8x8 = 64 patches, each 4x4x3=48 dims
        # Or: 2x2 patches = 16x16 = 256 patches... 
        # Let's do: resize to 28x28, then 2x2 patches = 14x14 = 196 patches!
        
        self.patch_size = 2
        self.img_size = 28  # resize to 28 so 28/2=14 patches per side
        self.num_patches = (self.img_size // self.patch_size) ** 2  # 14*14=196
        self.patch_dim = 3 * self.patch_size * self.patch_size  # 3*2*2=12
        
        # Patch embedding
        self.patch_embed = nn.Linear(self.patch_dim, self.d_model * 2)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, self.d_model * 2) * 0.02)
        
        # Use HormonicFormer backbone
        # Modify config for our sequence length
        config['model']['seq_len'] = self.seq_len
        self.backbone = HormonicFormerV7r3(config)
        
        # Classification head (reuse the classifier if exists, or create new)
        if hasattr(self.backbone, 'classifier') and self.backbone.classifier is not None:
            self.classifier = self.backbone.classifier
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.d_model * 2, self.d_model),
                nn.GELU(),
                nn.Dropout(config['model'].get('dropout', 0.1)),
                nn.Linear(self.d_model, 10)
            )
    
    def image_to_patches(self, images):
        """
        Convert images to patches
        images: [B, 3, 32, 32]
        return: [B, 196, 12] patches
        """
        B, C, H, W = images.shape
        
        # Resize to 28x28
        if H != self.img_size or W != self.img_size:
            images = F.interpolate(images, size=(self.img_size, self.img_size), 
                                  mode='bilinear', align_corners=False)
        
        # Reshape to patches
        # [B, C, H, W] -> [B, C, H//p, p, W//p, p] -> [B, H//p, W//p, C, p, p]
        p = self.patch_size
        patches = images.view(B, C, self.img_size // p, p, self.img_size // p, p)
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        patches = patches.view(B, self.num_patches, self.patch_dim)
        
        return patches
    
    def forward(self, images, targets=None):
        """
        images: [B, 3, 32, 32]
        targets: [B] class labels
        """
        B = images.size(0)
        
        # Convert to patches
        patches = self.image_to_patches(images)  # [B, 196, 12]
        
        # Embed patches
        x = self.patch_embed(patches)  # [B, 196, 128]
        x = x + self.pos_embed
        
        # Reshape to [B, S, D, 2] for HormonicFormer
        D = self.d_model
        psi = x.view(B, self.seq_len, D, 2)
        
        # Process through blocks manually (since we need to extract features)
        psi_list = []
        for i, block in enumerate(self.backbone.blocks):
            psi = block(psi)
            psi_list.append(psi)
            
            # PAC
            if self.backbone.pac and i > 0:
                amp = torch.sqrt(psi[...,0]**2 + psi[...,1]**2)
                mod = self.backbone.pac(psi_list[i-1], amp, i)
                ph = torch.atan2(psi[...,1], psi[...,0] + 1e-8)
                psi = torch.stack([mod * torch.cos(ph), mod * torch.sin(ph)], dim=-1)
        
        # Global average pooling
        h = self.backbone.output_norm(psi.view(B, self.seq_len, D * 2))
        pooled = h.mean(dim=1)  # [B, D*2]
        
        # Classify
        logits = self.classifier(pooled)
        
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
            return logits, loss
        return logits
    
    def get_diagnostics(self):
        """Get diagnostic metrics from first block"""
        block = self.backbone.blocks[0]
        nm = block.neuromod
        
        alpha_val = block.cgl.alpha.item()
        limit_r = math.sqrt(max(alpha_val, 0.0)) if alpha_val > 0 else 0.0
        
        return {
            'limit_cycle_r': limit_r,
            'alpha': alpha_val,
            'DA': (nm.get_alpha_modulation() - 0.5) if nm else 0.0,
            'CB': nm.cb_state.item() if nm else 0.0,
            'STP_eff': nm.stp.get_efficacy().mean().item() if nm else 1.0,
            'G_sparsity': (block.hebbian.G.data == 0).float().mean().item(),
        }


# 测试
if __name__ == '__main__':
    import math
    
    config = {
        'model': {
            'd_model': 64,
            'n_layers': 4,
            'n_heads': 4,
            'dropout': 0.1,
            'n_cgl_steps': 10,
            'D0_amp': 0.002,
            'D0_phase': 0.002,
            'cgl_dt': 0.02,
            'noise_scale': 0.001,
        },
        'use_neuromod': True,
        'use_pac': True,
        'use_pc': False,  # PC not needed for classification
        'g_coupling_strength': 0.1,
        'neuromod': {
            'da_init': 2.5,
            'da_ema_alpha': 0.9,
            'da_var_alpha': 0.9,
            'da_min': 0.1,
            'da_max': 0.9,
            'use_cb': True,
            'cb_gain': 2.0,
            'cb_threshold': 0.25,
            'tau_cb': 10.0,
            'cb_dt': 0.05,
        },
        'stp': {
            'U': 0.2,
            'tau_f': 1.0,
            'tau_d': 3.0,
            'dt': 0.05,
        },
        'hebbian': {
            'eta_potentiate': 0.001,
            'eta_depress': 0.0005,
            'sync_threshold': 0.3,
            'decay': 0.999,
        },
    }
    
    print("Testing HormonicCIFAR10...")
    model = HormonicCIFAR10(config)
    
    # Test forward
    images = torch.randn(4, 3, 32, 32)
    targets = torch.randint(0, 10, (4,))
    
    logits, loss = model(images, targets)
    print(f"  logits shape: {logits.shape}")
    print(f"  loss: {loss.item():.4f}")
    
    # Test diagnostics
    diag = model.get_diagnostics()
    print(f"  diagnostics: {diag}")
    
    print("\nOK - Ready for CIFAR-10 training!")
