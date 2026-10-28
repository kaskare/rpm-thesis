#!/bin/bash
# Draw all 8 graph representation permutations (granular x object_only x unified_graph).
# Saves to constellation/<example_num>/ by default.
# Run from project root.

SAMPLE="${1:-data/RAVEN-10000/left_center_single_right_center_single/RAVEN_0_train.npz}"

echo "Drawing all 8 permutations for $SAMPLE -> <config>/<example_num>/ (e.g. distribute_four/0/)"
echo ""

for granular in "" "--granular"; do
  for object_only in "" "--object-only"; do
    for unified_graph in "" "--unified-graph"; do
      FLAGS="$granular $object_only $unified_graph"
      echo "  granular=$([ -n "$granular" ] && echo true || echo false) object_only=$([ -n "$object_only" ] && echo true || echo false) unified=$([ -n "$unified_graph" ] && echo true || echo false)"
      python torch_gnn/draw_gnn_graph.py --sample "$SAMPLE" --use-gt $FLAGS
    done
  done
done

echo ""
echo "Done. All 8 graphs saved to <config>/<example_num>/"
