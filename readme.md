# Design of a Vector Embedding for Capability Composition
**PCCST503 – Machine Learning | Assignment 2**  
**Submitted by:** Devanandan J Y | Government Engineering College (GEC) Thrissur  

## Executive Summary
This repository delivers a mathematically rigorous Vector Embedding Architecture designed for autonomous capability composition. While traditional NLP embeddings (Word2Vec) capture semantic proximity, this system embeds functional operations into a continuous Cartesian space. By partitioning the vector space into discrete functional, schema, and operational subspaces, the architecture computationally verifies dependency chaining, identifies logical contradictions, and computes exact algebraic capability composition without requiring natural language inference.

## 1. Problem Definition
In Assignment 1, planning algorithms operated over pre-defined graph transitions in a state space $\mathbb{R}^d$. Real-world software systems, however, are driven by executable capabilities—modular operations constrained by prerequisites, operational latency, and resource schemas. The objective is to formulate an embedding function $\phi_C: \mathcal{C} \rightarrow \mathbb{R}^{d_c}$ where geometric alignment guarantees operational compatibility, allowing automated planners to orchestrate complex service pipelines purely via linear algebra.

## 2. Design Requirements
The proposed embedding architecture satisfies the following operational requirements:
1. **Capability Identity:** Unique vectors for distinct operations.
2. **Precondition-Effect Strictness:** Directional compatibility must reject logical contradictions (e.g., `DatabaseConfigured = False` blocking `DeploySupabase = True`).
3. **Similarity vs. Composability Decoupling:** Similar operations (e.g., Vercel Deployment vs. Render Deployment) must group together geometrically without falsely signaling sequential composability.
4. **Operational Homomorphism:** Cost and time must accumulate additively; probabilistic reliability must map to an additive vector space.

## 3. Related Embedding Approaches
Existing paradigms like TransE represent operations as translations ($h + r \approx t$). This captures net-delta state changes but entirely fails to encode static prerequisites (where a condition is required but not altered). Text-based LLM embeddings (e.g., APIBench) conflate description semantics with execution constraints, failing to provide the deterministic guarantees required for safe orchestrator planning.

## 4. Proposed Representation Architecture
To resolve the null-value ambiguity present in flat vectors, this implementation introduces a **Masked Subspace Architecture**. The capability vector $\phi_C(C_i)$ is partitioned into orthogonal sectors:

$$\phi_C(C_i) = [ \vec{p}_{val} \parallel \vec{p}_{mask} \parallel \vec{e}_{val} \parallel \vec{e}_{mask} \parallel \vec{q}_{ops} ]$$

*   $\vec{p}_{val}, \vec{e}_{val} \in \mathbb{R}^n$: Target values (1.0 for True, -1.0 for False) for preconditions and effects.
*   $\vec{p}_{mask}, \vec{e}_{mask} \in \mathbb{R}^n$: Binary masks indicating whether a variable is actively constrained or mutated (1.0) or ignored (0.0).
*   $\vec{q}_{ops} \in \mathbb{R}^3$: Operational vector containing Latency, Monetary Cost, and Log-Reliability.

## 5. Mathematical Formulation
**Log-Reliability Transformation:**
Execution reliabilities are probabilistic and multiplicative: $Rel_{12} = Rel_1 \times Rel_2$. To satisfy vector space additivity, we apply a negative log-likelihood transformation to the operational subspace:
$$\vec{q}[2] = -\ln(\max(Rel_i, 10^{-6}))$$
By logarithmic identity, $-\ln(Rel_1 \times Rel_2) = -\ln(Rel_1) + -\ln(Rel_2)$. Therefore, vector addition natively computes the composite failure probability exactly.

**Directed Compatibility:**
For transition $C_1 \rightarrow C_2$, the overlap mask is defined as $\vec{m}_{overlap} = \vec{e}_{mask,1} \odot \vec{p}_{mask,2}$. A strict contradiction exists if:
$$\sum ( \vec{m}_{overlap} \odot \mathbb{I}(\vert{}\vec{e}_{val,1} - \vec{p}_{val,2}\vert{} > \epsilon) ) > 0$$

## 6. Capability Composition Model
Given $C_{12} = C_2 \circ C_1$, the algebraic composition vector $\vec{v}_{12}$ is computed directly via element-wise operations on the masked subspaces:
*   **Composite Precondition Mask:** $\vec{p}_{mask,12} = \text{clip}(\vec{p}_{mask,1} + \vec{p}_{mask,2} \odot (1 - \vec{e}_{mask,1}), 0, 1)$
*   **Composite Effect Mask:** $\vec{e}_{mask,12} = \text{clip}(\vec{e}_{mask,2} + \vec{e}_{mask,1}, 0, 1)$

## 7. Implementation
The engine is implemented in Python `3.10+` using NumPy for highly optimized matrix operations. The `VectorSpace` class handles dynamic vocabulary mapping, masked array generation, and spatial similarity calculations.

## 8. Experimental Methodology & Results
The architecture was benchmarked against a Full-Stack Web Deployment pipeline (Next.js client, Supabase backend).

**Experiment 1: Capability Compatibility**
*   $C_1$ (InitSupabase) outputs `DatabaseReady = True`.
*   $C_2$ (DeployNextjs) requires `DatabaseReady = True`.
*   *Result:* Compatibility Score = `1.0`. Vector alignment perfectly validated the dependency chain.

**Experiment 2: Algebraic Composition**
Composing $C_2 \circ C_1$ yielded $\vec{v}_{12}$.
*   *Result:* $\vec{v}_{12}$ retained the `GitRepoExists = True` precondition from $C_1$, outputted `AppLive = True` from $C_2$, and exactly summed the latency subspace ($450ms + 1200ms = 1650ms$). Homomorphism holds with $\Delta = 0.0$.

**Experiment 3: Alternative Implementations**
*   $C_{vercel}$ (Deploy to Vercel) vs. $C_{render}$ (Deploy to Render). Both require a Git repository and result in `AppLive`.
*   *Result:* Functional Cosine Similarity = `1.0`. The system successfully recognizes them as functionally identical substitutes, despite divergent internal execution metadata.

**Experiment 4: Irrelevant Capabilities**
*   Comparing a distractor capability (e.g., `InstallSteamGame`) against the goal vector `AppLive = True`.
*   *Result:* Relevance Score = `0.0`. The dot product of the effect vector and goal vector instantly filtered the distractor.

**Experiment 5: Operational Attributes**
*   *Result:* Log-reliability addition yielded $-\ln(0.99) + -\ln(0.95) = 0.0613$. The inverse $e^{-0.0613} = 0.9405$ exactly matches the scalar probability multiplication ($0.99 \times 0.95$), proving vector integrity.

## 9. Comprehensive Analysis
*   **State Awareness:** The dual-mask architecture entirely resolves the limitation of standard vectors, allowing the planner to distinguish between "Must be False" ($\vec{p}_{val}=-1.0, \vec{p}_{mask}=1.0$) and "Irrelevant" ($\vec{p}_{val}=0.0, \vec{p}_{mask}=0.0$).
*   **Efficiency:** Matrix operations in $\mathbb{R}^{4n+3}$ scale linearly $\mathcal{O}(N)$ and execute in microseconds, making this suitable for high-frequency runtime replanning.

## 10. Limitations & Conclusion
**Limitations:** The current model assumes a fixed, global state vocabulary. Dynamically adding new state variables requires rebuilding the vector dimensions. Furthermore, strictly sequential composition is modeled; parallel pipeline execution (where latency is $\max(T_1, T_2)$ rather than $T_1 + T_2$) is not currently captured.

**Conclusion:** The Masked Subspace Architecture provides a theoretically sound, computationally efficient embedding for software capabilities. By mathematically decoupling functional transformations from operational costs, this design proves that complex application orchestration can be reduced to robust geometric vector algebra.

## 11. How to Run the Engine

The implementation is self-contained and requires Python 3.10+ with the `numpy` library. Follow these steps to execute the benchmark experiments:

**1. Install Dependencies**
Open your terminal and install the required matrix computation library:
```bash
pip install numpy