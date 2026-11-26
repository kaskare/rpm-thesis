# Notes: Šourek, Železný, Kuželka (2021), "Beyond Graph Neural Networks with Lifted Relational Neural Networks"

Machine Learning 110, 1695–1738 (arXiv 2007.06286). First author = Gustav Šír (published as Šourek), the
thesis supervisor. Read 2026-10-28 from the arXiv LaTeX source. Section numbers below are the paper's.

## What the paper claims is unique to LRNNs / NeuraLogic
- **Relational, not propositional.** "Common differentiable programming frameworks, such as PyTorch or
  TensorFlow, are effectively propositional": the user assembles modules procedurally. In LRNNs a theorem
  prover first assembles the program from a declarative template ("meta-programming"), and only then is it
  evaluated and differentiated (§7, footnote).
- **The network is a logical derivation** (§3.2). Remove the weights, compute the least Herbrand model of
  template + example, and turn it into a computation graph. A different example gives a different graph,
  with weights shared "in a scheme determined by the lifted structure of the template". Node types:
  - **fact nodes:** the example's weighted facts;
  - **atom nodes:** ground atoms, combining their rules with g∨;
  - **rule nodes:** one per ground rule body, combining it with g∧;
  - **aggregation nodes:** one per ground head per rule, with g* (avg, max, …) over the groundings.

  The aggregation "corresponds to a certain quantification over free variables appearing solely in the
  rule's body".
- **No restriction to message passing along given edges** (§5.2). Layers can reason over logically derived
  entities instead of the input graph:
  - paths of length 2 (`edge2(U,W) :- edge(U,V), edge(V,W)`);
  - motifs such as triangles (`node(U,V,W) :- edge(U,V), edge(V,W), edge(W,U)`) and 6-rings;
  - edges that aggregate adjacent edges;
  - type-driven diffusion;
  - latent type hierarchies by a recursive rule (`isa(A,C) :- isa(A,B), isa(B,C)`).
- **Recursion** (§4.3). Recursive nets, `n(P) :- n(C1), n(C2), n(C3), parent(P,C1,C2,C3)`, and recurrent
  nets, `h(Y) :- W_f f(Y), W_h h(X), next(X,Y)`, unfold to whatever depth the example's structure has.
- **Declarative background knowledge** is merged into the template with the same expressiveness, including
  arithmetic predicates such as `edge_sim(N1,N2) :- similar(N1,N2,S), W: ≥(S, 0.3)`. Hypergraphs and
  heterogeneous graphs need no special treatment.
- **Conclusion:** capturing such architectures "in an exact manner … would not be possible with the other
  approaches", meaning DeepProbLog, neural theorem provers and Logic Tensor Networks (§7–8).

## Experiments
- **Correctness:** GCN, GraphSAGE and GIN as templates train like their PyG and DGL implementations on the
  NCI molecule datasets.
- **Speed:** without batching, on CPU, LRNNs train about 10× faster per epoch than PyG on small sparse
  graphs. The lifted compression gives about 3× on the GNN templates. With batching and large dimensions,
  PyG on a GPU wins. Ground networks can be exported to other backends.
- **Generalization:** more expressive templates (GIN with edges, graphlets, latent bonds) fit the training
  data better but are not consistently better on test data. Even GCN often wins.
- **Stated limitation:** attention, gating and LSTMs "do not yield themselves naturally to the LRNN
  semantics".

## What we can use
1. **For the thesis framing.**
   - "Weights shared in a scheme determined by the lifted structure" is the informal version of our theorem
     (notes §21.2: a template is blind to the constants it does not name). We formalize what the paper
     states in one sentence, and connect it to the data's symmetries (§22).
   - Aggregation as quantification over body-only variables is the exact mechanism of the column roles
     (§24). The middle columns enter through one rule whose panel variable is aggregated, so permuting them
     changes nothing.
2. **The generalization result supports our framing.** More expressiveness did not help on test data,
   which fits "the right bias, not more capacity". Our capacity sweep (§23.2) found the same.
3. **Recursion, the clearest NeuraLogic-only construct, fits I-RAVEN-X.**
   - A fold along the row, `acc(c_{j+1}) :- f(c_{j+1}), acc(c_j), next(c_j, c_{j+1})`, reuses one weight
     set at every step. That builds in a different symmetry from the roles: **shift along the row** (the
     same step everywhere), the shape of Progression and of a running sum (Arithmetic).
   - It is independent of row length. **Checked 2026-10-28** in PyNeuraLogic 0.9.2: a two-rule template
     trained on rows of 3–5 values computed the sum of rows of 8 and 10 to a mean error of 0.009 (sums
     0–90). The weights learned the exact recurrence.
   - This would give a template trained on one row length that runs on another. ARLC claims that kind of
     transfer; our named-column and role models cannot do it.
   - Candidate plan item: train on 3×10 and test on other lengths at the same value range (IBM's generator
     makes them).
4. **Derived relations and background knowledge, between "same relation in every row" and "rules
   given".** For example, derive "successor" or "same value" relations with arithmetic predicates inside the
   template, rather than writing whole rules as the symbolic model does. This is a prior on relations, the
   second axis of the plan.
5. **Lifted compression is the ground-network counterpart of symmetry.** It merges sub-networks that
   compute identical values, detected by random-value identity testing. Our runs must switch it off
   (`iso_value_compression=False`): with it on, the symbolic model's results were invalid.
6. **Pruning.** The appendix (network pruning) says pruned single-input chains "do not contribute to learning capacity". In
   PyNeuraLogic 0.9.2 the pruning also drops a non-identity transformation (tanh) of single-rule atoms,
   which changes the function. That is why `rel` and `ans` are written as two rules (comment in `build_facet_template`).
7. **Gating.** The paper lists gating as unnatural. Our copy gates do it with ELPRODUCT in 0.9.2 (notes
   §20.2, §21.6), including content-based gates that keep the row symmetry.
