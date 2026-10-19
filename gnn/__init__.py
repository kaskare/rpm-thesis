"""RAVEN GNN package: object-level VSA-GNN training for RPM."""

from .config import (
    CONFIG_POS_MAP,
    NUM_CANDIDATES,
    NUM_CONTEXT,
    NUM_NODE_KINDS,
    NUM_PANEL_GRID,
    NUM_POSITIONS,
    RELATIONS,
    REL_TO_ID,
)
from .data import (
    RavenGTDataset,
    RavenImageDataset,
    build_position_centers,
    build_position_centers_universal,
    universal_position_centers,
    collate_gt,
    collate_images,
    discover_xml_paths,
    fallback_position_centers,
    get_config_from_path,
    gt_to_dense,
    parse_ground_truth,
)
from .graph import build_candidate_graph, build_unified_graph, infer_relations, pack_graphs
from .model import (
    CandidateVsaGNN,
    NodeEncoder,
    VSAEncoder,
    compute_w_index,
    load_codebook_only,
    scatter_mean,
)
from .train import (
    build_packed_candidates_from_batch,
    build_packed_unified,
    extract_batch_objects,
    run_epoch,
    run_epoch_gt,
    to_device_packed,
)

# Backward compatibility
_get_config_from_path = get_config_from_path

__all__ = [
    "CONFIG_POS_MAP",
    "NUM_CANDIDATES",
    "NUM_CONTEXT",
    "NUM_NODE_KINDS",
    "NUM_PANEL_GRID",
    "NUM_POSITIONS",
    "RELATIONS",
    "REL_TO_ID",
    "RavenGTDataset",
    "RavenImageDataset",
    "build_candidate_graph",
    "build_packed_candidates_from_batch",
    "build_packed_unified",
    "build_unified_graph",
    "build_position_centers",
    "build_position_centers_universal",
    "universal_position_centers",
    "CandidateVsaGNN",
    "collate_gt",
    "collate_images",
    "compute_w_index",
    "discover_xml_paths",
    "load_codebook_only",
    "extract_batch_objects",
    "fallback_position_centers",
    "get_config_from_path",
    "_get_config_from_path",
    "gt_to_dense",
    "infer_relations",
    "pack_graphs",
    "parse_ground_truth",
    "run_epoch",
    "run_epoch_gt",
    "scatter_mean",
    "to_device_packed",
]
