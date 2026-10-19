"""Minimal PyNeuraLogic 0.9.2 transcription of RelGNLayer / CandidateVsaGNN."""
from neuralogic.core import Model, R, V, Settings, Transformation, Aggregation, Metadata
from neuralogic.nn import Adam, CrossEntropy

D = 32
RELATIONS = ["left_of", "right_of", "above", "below", "same_row", "same_col",
             "overlap", "same_pos_cross_panel", "attr_to_obj", "obj_to_attr"]

AVG = Metadata(aggregation=Aggregation.AVG)
IDENT = Metadata(transformation=Transformation.IDENTITY)

m = Model()

# ── NodeEncoder: h0(O) = sum of per-attribute embeddings (additive, like nn.Embedding sums)
m += (R.h0(V.O)[D,] <= (R.pos(V.O, V.P),   R.emb_pos(V.P)[D,]))
m += (R.h0(V.O)[D,] <= (R.color(V.O, V.C), R.emb_col(V.C)[D,]))
m += (R.h0(V.O)[D,] <= (R.shape(V.O, V.S), R.emb_shp(V.S)[D,]))
m += (R.h0(V.O)[D,] <= (R.size(V.O, V.Z),  R.emb_siz(V.Z)[D,]))
m += (R.h0(V.O)[D,] <= (R.panel(V.O, V.G), R.emb_pnl(V.G)[D,]))
m += (R.h0 / 1) | IDENT          # sum the five contributions, no squashing

# ── 3x RelGNLayer: one weighted rule per typed relation, mean-aggregated per destination
for prev, cur in [("h0", "h1"), ("h1", "h2"), ("h2", "h3")]:
    for rel in RELATIONS:
        m += (R.get(cur)(V.Dst)[D, D] <= (
                  R.get(rel)(V.Src, V.Dst),
                  R.get(prev)(V.Src)[D, D],
              )) | AVG
    m += (R.get(cur)(V.X)[D, D] <= R.get(prev)(V.X)) | AVG   # residual path
    m += (R.get(cur) / 1) | Metadata(transformation=Transformation.RELU)

# ── Readout: mean-pool context nodes and candidate nodes, then score the pair
m += (R.ctx[D,]  <= (R.is_context(V.O),   R.h3(V.O)[D, D])) | AVG
m += (R.cand[D,] <= (R.is_candidate(V.O), R.h3(V.O)[D, D])) | AVG
m += (R.ctx / 0)  | Metadata(transformation=Transformation.RELU)
m += (R.cand / 0) | Metadata(transformation=Transformation.RELU)

m += (R.score[1, D] <= (R.ctx[D, D], R.cand[D, D]))
m += (R.score / 0) | IDENT

m.build(Settings(optimizer=Adam(lr=0.001), error_function=CrossEntropy()))
print("BUILD OK")
print("weight tensors:", len(list(m.parameters())))
print(m)
