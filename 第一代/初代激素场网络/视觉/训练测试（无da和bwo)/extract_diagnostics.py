import torch
import sys
import os

# Add model path
sys.path.insert(0, r'C:\Users\MR\Desktop\Kimi_Agent_模型评估\hormonic_v3')

def extract_diagnostics(checkpoint_path):
    """Extract neuromodulation diagnostics from checkpoint"""
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get epoch info
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"\nCheckpoint Epoch: {epoch}")
    
    # Try to load model to get diagnostics
    try:
        from models.hormonicformer_v3 import HormonicFormer
        
        # Get config from checkpoint
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # Default config
            config = {
                'd_model': 128,
                'n_layers': 2,
                'n_heads': 4,
                'n_steps': 3,
                'dropout': 0.1,
                'use_neuromod': True,
                'use_stp': True,
                'use_bwo': True,
                'target_sparsity': 0.7,
                'seq_len': 16,  # 4x4 patches
            }
        
        model = HormonicFormer(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Get diagnostics
        diagnostics = model.get_diagnostics()
        
        print("\n" + "="*60)
        print("神经调质诊断数据")
        print("="*60)
        
        # STP Efficacy
        if 'stp_efficacy_mean' in diagnostics:
            print(f"STP 效能 (u·r): {diagnostics['stp_efficacy_mean']:.4f}")
        
        # Homeostatic Gain
        if 'homeo_gain_mean' in diagnostics:
            print(f"稳态增益: {diagnostics['homeo_gain_mean']:.4f} ± {diagnostics.get('homeo_gain_std', 0):.4f}")
        
        # DA
        if 'da' in diagnostics:
            print(f"DA (多巴胺): {diagnostics['da']:.4f}")
        
        # CB
        if 'cb' in diagnostics:
            print(f"CB (皮质醇): {diagnostics['cb']:.4f}")
        
        # G Sparsity
        if 'G_sparsity' in diagnostics:
            sparsity = diagnostics['G_sparsity']
            print(f"G 稀疏度: {sparsity:.2%} (剪枝 {(1-sparsity)*100:.1f}%)")
            print(f"G 存活: {(1-sparsity)*100:.2f}%")
        
        print("="*60)
        
        return diagnostics
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nCheckpoint keys:", list(checkpoint.keys()))
        return None

if __name__ == "__main__":
    # Check common checkpoint locations
    checkpoint_paths = [
        r'C:\Users\MR\Desktop\Kimi_Agent_模型评估\hormonic_v3\checkpoints\best.pt',
        r'C:\Users\MR\Desktop\Kimi_Agent_模型评估\hormonic_v3\checkpoints\latest.pt',
        r'C:\Users\MR\Desktop\Kimi_Agent_模型评估\hormonic_v3\checkpoints\epoch_22.pt',
    ]
    
    for path in checkpoint_paths:
        if os.path.exists(path):
            print(f"\n{'='*60}")
            extract_diagnostics(path)
            break
    else:
        print("No checkpoint found. Checked:")
        for p in checkpoint_paths:
            print(f"  - {p}")
