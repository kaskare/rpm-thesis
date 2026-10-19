#*----------------------------------------------------------------------------*
#* GNN Training Script for NVSA-RAVEN                                         *
#* Trains a GNN on top of NVSA frontend outputs (SceneProb or VSA)           *
#*----------------------------------------------------------------------------*

import argparse
import os
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Local imports
from const import NUMPOS, DIM_COLOR, DIM_SIZE, DIM_TYPE
from dataset_nvsa import Dataset
import env_nvsa_ext as env_nvsa_ext
from util.averagemeter import AverageMeter
from util.checkpath import check_paths, save_checkpoint

# NVSA imports
from nvsa.perception.resnet import resnet18
from nvsa.perception.metrics import hd_mult_frontend, generate_IM
from nvsa.reasoning.vsa_block_utils import block_discrete_codebook, block_continuous_codebook

# GNN imports
from gnn_model import (
    RAVENGGNN,
    MultiAttributeGNN,
    VSAContextGNN,
    EdgeBuilder,
    FeatureExtractor,
    GraphBuilder,
    scene_prob_to_graph_batch,
    vsa_to_graph_batch,
    MAX_SCENE_DIM,
    MAX_POSITION_DIM,
    MAX_NUMBER_DIM,
    CONSTELLATION_CONFIGS
)


# =============================================================================
# Codebook Generation (same as main_nvsa_marg.py)
# =============================================================================

def generate_nvsa_codebooks(args, rng):
    """Generate or load NVSA codebooks."""
    imfilename = os.path.join(args.resume, "codebooks.pt") if args.resume else ""
    
    if args.resume and os.path.isfile(imfilename):
        print("Loading predefined NVSA codebooks...")
        data = torch.load(imfilename, map_location='cpu')
        perception_cb = data["perception_cb"]
        perception_imdict = data["perception_imdict"]
        backend_cb_cont = data['backend_cb_cont'].to(args.device)
        backend_cb_discrete = data['backend_cb_discrete'].to(args.device)
    else:
        print("Generating new NVSA codebooks...")
        backend_cb_cont, _ = block_continuous_codebook(
            device=args.device, scene_dim=12, 
            d=args.nvsa_backend_d, k=args.nvsa_backend_k, rng=rng
        )
        backend_cb_discrete, _ = block_discrete_codebook(
            device=args.device, 
            d=args.nvsa_backend_d, k=args.nvsa_backend_k, rng=rng
        )
        perception_cb, perception_imdict = generate_IM(
            args.d, (DIM_TYPE, DIM_SIZE, DIM_COLOR, args.num_pos), rng
        )
    
    return perception_cb, perception_imdict, backend_cb_cont, backend_cb_discrete


def load_pretrained_frontend(checkpoint_dir, device, args):
    """
    Load pretrained NVSA frontend from a checkpoint directory.
    
    The checkpoint directory should contain:
    - model_best.pth.tar (or checkpoint.pth.tar): Contains ResNet18 and metric_fc weights
    - codebooks.pt: Contains VSA codebooks
    
    Args:
        checkpoint_dir: Path to ckpt/ directory (e.g., log/nvsa_test01/.../0/ckpt/)
        device: torch device
        args: Arguments containing model configuration
        
    Returns:
        frontend_model: Loaded ResNet18 model
        metric_fc: Loaded hd_mult_frontend
        codebooks: Dict with perception_cb, perception_imdict, backend_cb_cont, backend_cb_discrete
    """
    # Load codebooks
    codebook_path = os.path.join(checkpoint_dir, "codebooks.pt")
    if not os.path.isfile(codebook_path):
        raise FileNotFoundError(f"Codebooks not found at {codebook_path}")
    
    print(f"Loading codebooks from {codebook_path}")
    codebooks = torch.load(codebook_path, map_location='cpu')
    perception_cb = codebooks["perception_cb"]
    perception_imdict = codebooks["perception_imdict"]
    backend_cb_cont = codebooks['backend_cb_cont'].to(device)
    backend_cb_discrete = codebooks['backend_cb_discrete'].to(device)
    
    # Load model checkpoint
    model_path = os.path.join(checkpoint_dir, "model_best.pth.tar")
    if not os.path.isfile(model_path):
        model_path = os.path.join(checkpoint_dir, "checkpoint.pth.tar")
    
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_dir}")
    
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', '?')} with acc {checkpoint.get('best_acc', '?')}%")
    
    # Initialize models
    # IMPORTANT: pretrained=True sets in_channels=3 (architecture matches saved checkpoint)
    # We're not loading ImageNet weights here, just creating matching architecture
    # The actual weights come from the checkpoint below
    frontend_model = resnet18(
        pretrained=True,  # Creates 3-channel conv1 to match saved checkpoint architecture
        progress=False,   # Don't download ImageNet weights (we'll overwrite anyway)
        num_classes=args.d,
        no_maxpool=args.no_maxpool
    )
    
    metric_fc = hd_mult_frontend(
        num_features=args.d,
        num_classes=5,
        mat=perception_cb,
        s=args.s,
        trainable_s=args.trainable_s,
        marg_m=args.m,
        marg_in_act=args.in_act,
        num_pos=args.num_pos,
        fixed_weights=1
    )
    
    # Load weights
    frontend_model.load_state_dict(checkpoint['state_dict_model'])
    metric_fc.load_state_dict(checkpoint['state_dict_metric_fc'])
    
    # Move to device
    frontend_model.to(device)
    metric_fc.to(device)
    
    return frontend_model, metric_fc, {
        'perception_cb': perception_cb,
        'perception_imdict': perception_imdict,
        'backend_cb_cont': backend_cb_cont,
        'backend_cb_discrete': backend_cb_discrete
    }


def get_pretrained_checkpoint_path(log_dir, config):
    """
    Find the checkpoint path for a specific constellation in a log directory.
    
    Args:
        log_dir: Base log directory (e.g., log/nvsa_test01/)
        config: Constellation name (e.g., 'center_single')
        
    Returns:
        Path to the ckpt/ directory
    """
    import glob
    
    # Find directories matching the config
    pattern = os.path.join(log_dir, f"{config}_*", "0", "ckpt")
    matches = glob.glob(pattern)
    
    if not matches:
        raise FileNotFoundError(f"No checkpoint found for config '{config}' in {log_dir}")
    
    if len(matches) > 1:
        print(f"Warning: Multiple checkpoints found for '{config}', using first one:")
        for m in matches:
            print(f"  {m}")
    
    return matches[0]


# =============================================================================
# Feature Extraction Functions
# =============================================================================

def extract_scene_prob_features(
    scene_prob,
    feature_mode: str = "concat",
    pad_to_max: bool = True
) -> torch.Tensor:
    """
    Extract features from SceneProb.
    
    Args:
        scene_prob: SceneProb namedtuple from env.prepare()
        feature_mode: "concat", "fixed", or specific attribute
        pad_to_max: Whether to pad to maximum dimension
        
    Returns:
        features: [batch, 16, feature_dim]
    """
    return FeatureExtractor.scene_prob_to_tensor(
        scene_prob, 
        mode=feature_mode,
        pad_to_max=pad_to_max
    )


def extract_vsa_features(
    scene_prob,
    scene_logprob,
    env,
    all_panels: bool = True
) -> dict:
    """
    Extract VSA backend features.
    
    Args:
        scene_prob: SceneProb from env.prepare()
        scene_logprob: SceneLogProb from env.prepare()
        env: NVSA environment
        all_panels: If True, compute VSA for all 16 panels, else only 8 context
        
    Returns:
        Dictionary of VSA vectors per attribute
    """
    vsa_features = {}
    
    # Access planners from environment
    # Note: Different constellations have different planner structures
    if hasattr(env, 'planner'):
        # Single constellation (center_single, distribute_four, etc.)
        planners = {'main': env.planner}
    elif hasattr(env, 'planner1') and hasattr(env, 'planner2'):
        # Composite constellation
        planners = {'inner': env.planner1, 'outer': env.planner2}
    else:
        raise ValueError("Could not find planners in environment")
    
    # For simplicity, use the first planner's codebooks
    planner = list(planners.values())[0]
    
    # Get attribute-specific planners
    attr_planners = {
        'position': planner.position_planner if hasattr(planner, 'position_planner') else None,
        'number': planner.number_planner if hasattr(planner, 'number_planner') else None,
        'type': planner.type_planner if hasattr(planner, 'type_planner') else None,
        'size': planner.size_planner if hasattr(planner, 'size_planner') else None,
        'color': planner.color_planner if hasattr(planner, 'color_planner') else None,
    }
    
    # Compute VSA vectors for each attribute
    for attr_name, attr_planner in attr_planners.items():
        if attr_planner is None:
            continue
            
        # Get the appropriate probability
        logprob = getattr(scene_logprob, f"{attr_name}_logprob")
        prob = torch.exp(logprob)
        
        # Select panels
        if all_panels:
            panels_prob = prob  # [batch, 16, attr_dim]
        else:
            panels_prob = prob[:, :8]  # [batch, 8, attr_dim] - context only
        
        # Get codebook
        if hasattr(attr_planner, 'vsa_cb_discrete_c'):
            cb = attr_planner.vsa_cb_discrete_c
        elif hasattr(attr_planner, 'vsa_cb_cont_c'):
            cb = attr_planner.vsa_cb_cont_c
        else:
            continue
        
        # Compute VSA vectors using pmf2vec
        # Need to ensure dimensions match
        scene_dim = cb.shape[0]
        if panels_prob.shape[-1] > scene_dim:
            panels_prob = panels_prob[..., :scene_dim]
        elif panels_prob.shape[-1] < scene_dim:
            pad_size = scene_dim - panels_prob.shape[-1]
            panels_prob = F.pad(panels_prob, (0, pad_size), value=0)
        
        from nvsa.reasoning.vsa_block_utils import pmf2vec
        p_vsa = pmf2vec(cb, panels_prob)  # [batch, num_panels, k, L]
        
        vsa_features[attr_name] = p_vsa
    
    return vsa_features


# =============================================================================
# GNN Model Factory
# =============================================================================

def create_gnn_model(args) -> nn.Module:
    """Create GNN model based on arguments."""
    
    if args.feature_type == "scene_prob":
        if args.feature_mode == "concat":
            input_dim = MAX_SCENE_DIM if args.pad_features else CONSTELLATION_CONFIGS[args.config].total_scene_dim
        elif args.feature_mode == "fixed":
            input_dim = DIM_TYPE + 1 + DIM_SIZE + 1 + DIM_COLOR + 1  # 24
        else:
            raise ValueError(f"Unknown feature_mode: {args.feature_mode}")
        
        model = RAVENGGNN(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_gnn_layers,
            conv_type=args.conv_type,
            dropout=args.dropout,
            num_candidates=8,
            use_projection=True,
            pool_type=args.pool_type
        )
    
    elif args.feature_type == "vsa":
        if args.gnn_model == "multi_attr":
            model = MultiAttributeGNN(
                hidden_dim=args.hidden_dim,
                num_layers=args.num_gnn_layers,
                conv_type=args.conv_type,
                dropout=args.dropout,
                num_candidates=8,
                vsa_dim=args.nvsa_backend_d,
                use_vsa=True
            )
        else:
            # Single GNN on concatenated VSA
            input_dim = args.nvsa_backend_d * 5  # 5 attributes
            model = RAVENGGNN(
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_gnn_layers,
                conv_type=args.conv_type,
                dropout=args.dropout,
                num_candidates=8,
                use_projection=True,
                pool_type=args.pool_type
            )
    
    elif args.feature_type == "vsa_context":
        model = VSAContextGNN(
            input_dim=args.nvsa_backend_d,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_gnn_layers,
            conv_type=args.conv_type,
            dropout=args.dropout,
            output_type="embedding"
        )
    
    else:
        raise ValueError(f"Unknown feature_type: {args.feature_type}")
    
    return model


# =============================================================================
# Training Functions
# =============================================================================

def train_gnn(args):
    """Main training function."""
    
    device = args.device
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    
    # Setup logging
    writer = SummaryWriter(args.log_dir)
    
    # Initialize RNG for codebook generation
    rng = np.random.default_rng(seed=args.seed)
    
    # Determine frontend checkpoint path
    frontend_ckpt_dir = None
    if args.frontend_checkpoint:
        frontend_ckpt_dir = args.frontend_checkpoint
    elif args.nvsa_log_dir:
        # Auto-find checkpoint for this constellation
        frontend_ckpt_dir = get_pretrained_checkpoint_path(args.nvsa_log_dir, args.config)
        print(f"Auto-detected frontend checkpoint: {frontend_ckpt_dir}")
    
    # Load pretrained frontend or generate new codebooks
    if frontend_ckpt_dir:
        print(f"Loading pretrained frontend from {frontend_ckpt_dir}")
        frontend_model, metric_fc, codebooks = load_pretrained_frontend(
            frontend_ckpt_dir, device, args
        )
        perception_cb = codebooks['perception_cb']
        backend_cb_cont = codebooks['backend_cb_cont']
        backend_cb_discrete = codebooks['backend_cb_discrete']
    else:
        # Generate/load codebooks
        perception_cb, perception_imdict, backend_cb_cont, backend_cb_discrete = generate_nvsa_codebooks(args, rng)
        
        # Initialize frontend models from scratch
        print("Initializing frontend models from scratch...")
        frontend_model = resnet18(
            pretrained=args.pretrained,
            progress=True,
            num_classes=args.d,
            no_maxpool=args.no_maxpool
        )
        
        metric_fc = hd_mult_frontend(
            num_features=args.d,
            num_classes=5,
            mat=perception_cb,
            s=args.s,
            trainable_s=args.trainable_s,
            marg_m=args.m,
            marg_in_act=args.in_act,
            num_pos=args.num_pos,
            fixed_weights=1
        )
        
        frontend_model.to(device)
        metric_fc.to(device)
    
    # Create NVSA environment
    env = env_nvsa_ext.get_env(
        args.config, device,
        vsa_cb_discrete=backend_cb_discrete,
        vsa_cb_cont=backend_cb_cont,
        detector_m=args.detector_m,
        detector_s=1,
        detector_act=args.detector_act,
        executor_m=0,
        executor_s=args.executor_s,
        executor_act=args.executor_act,
        executor_cos2pmf_act=args.executor_cos2pmf_act
    )
    
    # Freeze frontend if specified
    if args.freeze_frontend:
        print("Freezing frontend weights...")
        for param in frontend_model.parameters():
            param.requires_grad = False
        for param in metric_fc.parameters():
            param.requires_grad = False
        frontend_model.eval()
        metric_fc.eval()
    
    # Create GNN model
    print("Creating GNN model...")
    gnn_model = create_gnn_model(args)
    gnn_model.to(device)
    
    # Pre-build edge index
    edge_index = EdgeBuilder.build_full_panel_edges(
        bidirectional=True,
        include_answer_edges=True
    ).to(device)
    
    # Setup optimizer
    train_params = list(gnn_model.parameters())
    if not args.freeze_frontend:
        train_params += list(frontend_model.parameters())
        train_params += list(metric_fc.parameters())
    
    optimizer = optim.Adam(train_params, lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_acc = 0
    
    if args.gnn_checkpoint and os.path.isfile(args.gnn_checkpoint):
        print(f"Loading GNN checkpoint from {args.gnn_checkpoint}")
        checkpoint = torch.load(args.gnn_checkpoint, map_location=device)
        gnn_model.load_state_dict(checkpoint['state_dict_gnn'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint.get('best_acc', 0)
    
    # Create data loaders
    print("Creating data loaders...")
    train_set = Dataset(
        args.dataset, "train", args.config,
        gen_attribute=args.gen_attribute,
        gen_rule=args.gen_rule
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_set = Dataset(
        args.dataset, "val", args.config,
        test=True,
        gen_attribute=args.gen_attribute,
        gen_rule=args.gen_rule
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Feature type: {args.feature_type}")
    print(f"Feature mode: {args.feature_mode}")
    print(f"GNN: {args.conv_type} with {args.num_gnn_layers} layers, hidden_dim={args.hidden_dim}")
    print(f"Frontend frozen: {args.freeze_frontend}")
    print("-" * 60)
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            args, epoch,
            train_loader, 
            frontend_model, metric_fc, gnn_model, env,
            edge_index, optimizer, criterion, device, writer
        )
        
        # Validate
        with torch.no_grad():
            val_loss, val_acc = validate_epoch(
                args, epoch,
                val_loader,
                frontend_model, metric_fc, gnn_model, env,
                edge_index, criterion, device, writer
            )
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        
        save_dict = {
            'epoch': epoch + 1,
            'state_dict_gnn': gnn_model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'args': vars(args)
        }
        
        if not args.freeze_frontend:
            save_dict['state_dict_model'] = frontend_model.state_dict()
            save_dict['state_dict_metric_fc'] = metric_fc.state_dict()
        
        save_checkpoint(save_dict, is_best, savedir=args.checkpoint_dir)
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}% {'(BEST)' if is_best else ''}")
    
    print(f"\nTraining completed! Best validation accuracy: {best_acc:.2f}%")
    
    # Final test
    if args.test_after_train:
        print("\nRunning final test...")
        test_gnn(args, gnn_model, frontend_model, metric_fc, env, edge_index, device, writer)
    
    writer.close()
    return best_acc


def train_epoch(
    args, epoch,
    data_loader,
    frontend_model, metric_fc, gnn_model, env,
    edge_index, optimizer, criterion, device, writer
):
    """Train for one epoch."""
    
    if not args.freeze_frontend:
        frontend_model.train()
        metric_fc.train()
    gnn_model.train()
    
    loss_avg = AverageMeter('Loss', ':.4f')
    acc_avg = AverageMeter('Acc', ':.2f')
    
    for batch_idx, (images, targets, _) in enumerate(tqdm(data_loader, desc=f"Train Epoch {epoch}")):
        images = images.to(device)
        targets = targets.to(device)
        
        B, N, _, H, W = images.shape
        
        # Forward through frontend
        if args.freeze_frontend:
            with torch.no_grad():
                model_output = frontend_model(images.view(B * N, 1, H, W))
                marg_prob = metric_fc(model_output, B, N)
                scene_prob, scene_logprob = env.prepare(marg_prob)
        else:
            model_output = frontend_model(images.view(B * N, 1, H, W))
            marg_prob = metric_fc(model_output, B, N)
            scene_prob, scene_logprob = env.prepare(marg_prob)
        
        # Extract features for GNN
        if args.feature_type == "scene_prob":
            features = extract_scene_prob_features(
                scene_prob,
                feature_mode=args.feature_mode,
                pad_to_max=args.pad_features
            )
            # Forward through GNN
            scores = gnn_model(features, edge_index)
        
        elif args.feature_type == "vsa":
            # For VSA features, we need a simpler approach
            # Concatenate all SceneProb features and use them
            # (Full VSA extraction requires more complex planner access)
            features = extract_scene_prob_features(
                scene_prob,
                feature_mode="concat",
                pad_to_max=args.pad_features
            )
            scores = gnn_model(features, edge_index)
        
        else:
            raise ValueError(f"Unknown feature_type: {args.feature_type}")
        
        # Compute loss
        loss = criterion(scores, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), args.clip)
        
        optimizer.step()
        
        # Compute accuracy
        pred = scores.argmax(dim=-1)
        acc = (pred == targets).float().mean().item() * 100
        
        loss_avg.update(loss.item(), B)
        acc_avg.update(acc, B)
    
    # Log to tensorboard
    writer.add_scalar('train/loss', loss_avg.avg, epoch)
    writer.add_scalar('train/acc', acc_avg.avg, epoch)
    
    return loss_avg.avg, acc_avg.avg


def validate_epoch(
    args, epoch,
    data_loader,
    frontend_model, metric_fc, gnn_model, env,
    edge_index, criterion, device, writer
):
    """Validate for one epoch."""
    
    frontend_model.eval()
    metric_fc.eval()
    gnn_model.eval()
    
    loss_avg = AverageMeter('Loss', ':.4f')
    acc_avg = AverageMeter('Acc', ':.2f')
    
    for batch_idx, (images, targets, _) in enumerate(tqdm(data_loader, desc=f"Val Epoch {epoch}")):
        images = images.to(device)
        targets = targets.to(device)
        
        B, N, _, H, W = images.shape
        
        # Forward through frontend
        model_output = frontend_model(images.view(B * N, 1, H, W))
        marg_prob = metric_fc(model_output, B, N)
        scene_prob, scene_logprob = env.prepare(marg_prob)
        
        # Extract features for GNN
        if args.feature_type == "scene_prob":
            features = extract_scene_prob_features(
                scene_prob,
                feature_mode=args.feature_mode,
                pad_to_max=args.pad_features
            )
        else:
            features = extract_scene_prob_features(
                scene_prob,
                feature_mode="concat",
                pad_to_max=args.pad_features
            )
        
        # Forward through GNN
        scores = gnn_model(features, edge_index)
        
        # Compute loss
        loss = criterion(scores, targets)
        
        # Compute accuracy
        pred = scores.argmax(dim=-1)
        acc = (pred == targets).float().mean().item() * 100
        
        loss_avg.update(loss.item(), B)
        acc_avg.update(acc, B)
    
    # Log to tensorboard
    writer.add_scalar('val/loss', loss_avg.avg, epoch)
    writer.add_scalar('val/acc', acc_avg.avg, epoch)
    
    return loss_avg.avg, acc_avg.avg


def test_gnn(args, gnn_model, frontend_model, metric_fc, env, edge_index, device, writer=None):
    """Test the GNN model."""
    
    frontend_model.eval()
    metric_fc.eval()
    gnn_model.eval()
    
    test_set = Dataset(
        args.dataset, "test", args.config,
        test=True,
        gen_attribute=args.gen_attribute,
        gen_rule=args.gen_rule
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    acc_avg = AverageMeter('Acc', ':.2f')
    
    with torch.no_grad():
        for batch_idx, (images, targets, _) in enumerate(tqdm(test_loader, desc="Testing")):
            images = images.to(device)
            targets = targets.to(device)
            
            B, N, _, H, W = images.shape
            
            # Forward through frontend
            model_output = frontend_model(images.view(B * N, 1, H, W))
            marg_prob = metric_fc(model_output, B, N)
            scene_prob, scene_logprob = env.prepare(marg_prob)
            
            # Extract features
            features = extract_scene_prob_features(
                scene_prob,
                feature_mode=args.feature_mode,
                pad_to_max=args.pad_features
            )
            
            # Forward through GNN
            scores = gnn_model(features, edge_index)
            
            # Compute accuracy
            pred = scores.argmax(dim=-1)
            acc = (pred == targets).float().mean().item() * 100
            acc_avg.update(acc, B)
    
    print(f"Test Accuracy: {acc_avg.avg:.2f}%")
    
    if writer:
        writer.add_scalar('test/acc', acc_avg.avg, 0)
    
    # Save results
    np.savez(
        os.path.join(args.save_dir, f"test_result_{args.config}.npz"),
        acc=acc_avg.avg
    )
    
    return acc_avg.avg


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='GNN Training on NVSA-RAVEN')
    
    # Mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Training or testing mode')
    
    # Paths
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to RAVEN/I-RAVEN dataset')
    parser.add_argument('--exp-dir', type=str, default='./experiments/gnn/',
                        help='Experiment directory')
    parser.add_argument('--frontend-checkpoint', type=str, default='',
                        help='Path to pretrained frontend ckpt/ directory')
    parser.add_argument('--nvsa-log-dir', type=str, default='',
                        help='Path to NVSA log directory (e.g., log/nvsa_test01/) to auto-find checkpoint')
    parser.add_argument('--gnn-checkpoint', type=str, default='',
                        help='Path to GNN checkpoint for resuming')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to resume directory (for codebooks)')
    
    # Dataset
    parser.add_argument('--config', type=str, default='center_single',
                        choices=list(NUMPOS.keys()),
                        help='RPM constellation configuration')
    parser.add_argument('--gen-attribute', type=str, default='',
                        help='OOD generalization attribute')
    parser.add_argument('--gen-rule', type=str, default='',
                        help='OOD generalization rule')
    
    # GNN Architecture
    parser.add_argument('--feature-type', type=str, default='scene_prob',
                        choices=['scene_prob', 'vsa', 'vsa_context'],
                        help='Type of features for GNN')
    parser.add_argument('--feature-mode', type=str, default='concat',
                        choices=['concat', 'fixed'],
                        help='Feature mode for scene_prob')
    parser.add_argument('--pad-features', action='store_true', default=True,
                        help='Pad features to max dimension')
    parser.add_argument('--gnn-model', type=str, default='single',
                        choices=['single', 'multi_attr'],
                        help='GNN model type')
    parser.add_argument('--conv-type', type=str, default='gcn',
                        choices=['gcn', 'gat'],
                        help='Graph convolution type')
    parser.add_argument('--hidden-dim', type=int, default=512,
                        help='GNN hidden dimension')
    parser.add_argument('--num-gnn-layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--pool-type', type=str, default='context',
                        choices=['context', 'candidates', 'global'],
                        help='Pooling type for answer scoring')
    
    # Frontend Settings
    parser.add_argument('--freeze-frontend', action='store_true',
                        help='Freeze frontend weights')
    parser.add_argument('--d', type=int, default=512,
                        help='Frontend VSA dimension')
    parser.add_argument('--s', type=float, default=7.,
                        help='Inverse softmax temperature')
    parser.add_argument('--trainable-s', action='store_true',
                        help='Make s trainable')
    parser.add_argument('--m', type=float, default=0.,
                        help='Marginalization threshold')
    parser.add_argument('--in-act', type=str, default='ReLU',
                        help='Input activation')
    parser.add_argument('--pretrained', type=int, default=1,
                        help='Use pretrained ResNet')
    parser.add_argument('--no-maxpool', type=int, default=1,
                        help='No maxpool in ResNet')
    
    # NVSA Backend Settings
    parser.add_argument('--nvsa-backend-d', type=int, default=1024,
                        help='VSA backend dimension')
    parser.add_argument('--nvsa-backend-k', type=int, default=4,
                        help='Number of VSA blocks')
    parser.add_argument('--detector-act', type=str, default='threshold',
                        help='Detector activation')
    parser.add_argument('--detector-m', type=float, default=0.0,
                        help='Detector threshold')
    parser.add_argument('--executor-act', type=str, default='ReLU',
                        help='Executor activation')
    parser.add_argument('--executor-s', type=float, default=1,
                        help='Executor scaling')
    parser.add_argument('--executor-cos2pmf-act', type=str, default='Identity',
                        help='Executor cos2pmf activation')
    
    # Training Settings
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--lr-step', type=int, default=30,
                        help='LR scheduler step size')
    parser.add_argument('--lr-gamma', type=float, default=0.5,
                        help='LR scheduler gamma')
    parser.add_argument('--clip', type=float, default=1.0,
                        help='Gradient clipping')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device index')
    
    # Testing
    parser.add_argument('--test-after-train', action='store_true',
                        help='Run test after training')
    
    # Run ID
    parser.add_argument('--run', type=int, default=0,
                        help='Run ID for multiple experiments')
    
    args = parser.parse_args()
    
    # Setup device
    args.cuda = torch.cuda.is_available()
    args.device = torch.device(f"cuda:{args.device}" if args.cuda else "cpu")
    
    # Number of positions for constellation
    args.num_pos = NUMPOS[args.config]
    
    # Setup directories
    exp_name = f"{args.config}_{args.feature_type}_{args.feature_mode}_{args.conv_type}_h{args.hidden_dim}_l{args.num_gnn_layers}"
    if args.freeze_frontend:
        exp_name += "_frozen"
    exp_name += f"/run{args.run}"
    
    args.exp_dir = os.path.join(args.exp_dir, exp_name)
    args.checkpoint_dir = os.path.join(args.exp_dir, "ckpt")
    args.save_dir = os.path.join(args.exp_dir, "save")
    args.log_dir = os.path.join(args.exp_dir, "log")
    
    # Create directories
    for d in [args.checkpoint_dir, args.save_dir, args.log_dir]:
        os.makedirs(d, exist_ok=True)
    
    print("=" * 60)
    print("GNN Training for NVSA-RAVEN")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Feature type: {args.feature_type}")
    print(f"Feature mode: {args.feature_mode}")
    print(f"GNN: {args.conv_type}, layers={args.num_gnn_layers}, hidden={args.hidden_dim}")
    print(f"Freeze frontend: {args.freeze_frontend}")
    print(f"Device: {args.device}")
    print(f"Experiment dir: {args.exp_dir}")
    print("=" * 60)
    
    if args.mode == 'train':
        train_gnn(args)
    else:
        # Test mode
        if not args.gnn_checkpoint:
            args.gnn_checkpoint = os.path.join(args.checkpoint_dir, "model_best.pth.tar")
        
        # Load models and test
        rng = np.random.default_rng(seed=args.seed)
        perception_cb, perception_imdict, backend_cb_cont, backend_cb_discrete = generate_nvsa_codebooks(args, rng)
        
        env = env_nvsa_ext.get_env(
            args.config, args.device,
            vsa_cb_discrete=backend_cb_discrete,
            vsa_cb_cont=backend_cb_cont,
            detector_m=args.detector_m,
            detector_s=1,
            detector_act=args.detector_act,
            executor_m=0,
            executor_s=args.executor_s,
            executor_act=args.executor_act,
            executor_cos2pmf_act=args.executor_cos2pmf_act
        )
        
        frontend_model = resnet18(
            pretrained=args.pretrained,
            progress=True,
            num_classes=args.d,
            no_maxpool=args.no_maxpool
        )
        
        metric_fc = hd_mult_frontend(
            num_features=args.d,
            num_classes=5,
            mat=perception_cb,
            s=args.s,
            trainable_s=args.trainable_s,
            marg_m=args.m,
            marg_in_act=args.in_act,
            num_pos=args.num_pos,
            fixed_weights=1
        )
        
        gnn_model = create_gnn_model(args)
        
        # Load checkpoints
        checkpoint = torch.load(args.gnn_checkpoint, map_location=args.device)
        gnn_model.load_state_dict(checkpoint['state_dict_gnn'])
        
        if args.frontend_checkpoint:
            frontend_ckpt = torch.load(args.frontend_checkpoint, map_location=args.device)
            frontend_model.load_state_dict(frontend_ckpt['state_dict_model'])
            metric_fc.load_state_dict(frontend_ckpt['state_dict_metric_fc'])
        elif 'state_dict_model' in checkpoint:
            frontend_model.load_state_dict(checkpoint['state_dict_model'])
            metric_fc.load_state_dict(checkpoint['state_dict_metric_fc'])
        
        frontend_model.to(args.device)
        metric_fc.to(args.device)
        gnn_model.to(args.device)
        
        edge_index = EdgeBuilder.build_full_panel_edges(
            bidirectional=True,
            include_answer_edges=True
        ).to(args.device)
        
        test_gnn(args, gnn_model, frontend_model, metric_fc, env, edge_index, args.device)


if __name__ == "__main__":
    main()
