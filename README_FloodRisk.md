# Human-in-the-Loop Flood Risk Prediction for Urban Drainage Networks

A physics-informed machine learning pipeline for predicting and explaining flood risk across urban sewer and stormwater networks under climate uncertainty — built as exploratory groundwork for research into Human-in-the-Loop AI for climate-resilient infrastructure.

---

## Overview

Urban drainage systems face escalating failure risk from climate extremes and ageing infrastructure. Traditional hydraulic models are computationally slow and lack scalability, while opaque AI methods risk producing outputs that engineers cannot interpret or trust. This project addresses both problems by building a pipeline that is:

- **Physics-informed** — flood risk calculations embed hydraulic principles (Manning's equation, runoff physics)
- **Uncertainty-aware** — 500 climate scenarios capture the full range of rainfall variability including rare 1-in-100-year events
- **Explainable** — permutation importance identifies which physical and climatic factors drive risk at each node
- **Human-aligned** — outputs are designed as decision support for infrastructure engineers, not black-box predictions

The project directly maps onto the research agenda of the QUEX Institute joint PhD programme between the University of Exeter and the University of Queensland on Human-in-the-Loop AI for climate-resilient and equitable sewer systems.

---

## Motivation

This work is informed by the challenge identified in recent literature: generative AI and physics-informed ML can produce high-quality infrastructure predictions, but the remaining open problem is ensuring those predictions are interpretable and aligned with the values and judgement of human decision-makers — particularly in climate-critical contexts where the stakes of failure are high and marginalised communities are disproportionately affected.

The three pillars of this project correspond directly to the three research components of the PhD:

| PhD Component | This Project |
|---|---|
| Physics-informed ML | Hydraulic surrogate model with Manning's equation constraints |
| Scenario generation under uncertainty | 500 stochastic climate scenarios (rainfall intensity, duration, return period, climate factor) |
| Human-in-the-Loop AI | Explainability layer + practitioner-facing decision dashboard |

---

## Pipeline

```
Urban drainage network (graph: nodes = junctions, edges = pipes)
        ↓
Physical property assignment (elevation, pipe diameter, catchment area, impermeability, age)
        ↓
Climate scenario generation (500 scenarios: rainfall intensity, duration, return period, antecedent moisture, climate factor)
        ↓
Physics-informed flood risk simulation (hydraulic surrogate: Manning's equation proxy)
        ↓
Gradient Boosting surrogate ML model (learns the risk function from 75,000 node-scenario samples)
        ↓
Explainability analysis (permutation importance — which factors drive risk?)
        ↓
Human-in-the-Loop decision dashboard (network risk map, risk categories, climate sensitivity)
```

---

## Key Results

- Surrogate ML model achieves **R2 > 0.91** on held-out test data
- Feature importance reveals the relative contribution of rainfall intensity, pipe capacity, impermeability, and climate change factor to flood risk
- Decision dashboard identifies **critical and high-risk nodes** across all simulated scenarios
- Climate sensitivity analysis highlights which nodes are most vulnerable under future rainfall projections — a key input for infrastructure prioritisation decisions

---

## Outputs

| File | Description |
|---|---|
| `drainage_network.png` | Graph visualisation of the drainage network coloured by node elevation |
| `flood_risk_model.png` | Model accuracy and explainability — feature importance + predicted vs actual |
| `hitl_dashboard.png` | Human-in-the-Loop decision support dashboard — risk map, categories, climate sensitivity |

---

## Technical Approach

### Graph-based network representation
The drainage network is modelled as a graph where nodes represent pipe junctions and edges represent pipe segments. Each node stores physical properties. This representation directly mirrors the Graph Neural Network (GNN) surrogate digital twin approach used in state-of-the-art sewer system modelling, where GNNs embed hydraulic principles to propagate information across the pipe network topology.

### Physics-informed risk calculation
Flood risk at each node is computed using a simplified hydraulic model that encodes:
- Runoff volume = rainfall intensity x duration x catchment area x impermeability x climate factor
- Pipe capacity (Manning's equation proxy) = diameter^2.67 x hydraulic gradient^0.5
- Risk score incorporates antecedent soil moisture, storm rarity (return period), and infrastructure age

This is not a purely data-driven model — physical constraints are embedded directly into the risk function, ensuring outputs respect hydraulic conservation principles even under extrapolation to unseen climate scenarios.

### Scenario generation under uncertainty
500 climate scenarios are generated by sampling:
- Rainfall intensity from a Gamma distribution (matching observed rainfall statistics)
- Storm duration from an Exponential distribution
- Return period from {2, 5, 10, 25, 50, 100} year events
- Antecedent soil moisture uniformly across dry to saturated conditions
- Climate change multiplier from 1.0x (present) to 1.4x (future +40% intensification)

This stochastic approach captures deep uncertainty in future climate projections and allows risk to be expressed as a distribution rather than a single point estimate.

### Explainability for human decision-makers
Permutation importance is computed on the held-out test set to quantify which features most influence the surrogate model's predictions. This is the core of the Human-in-the-Loop component: infrastructure engineers need to understand not just what the AI predicts, but why — which physical factors are driving risk at each node, so they can interrogate, challenge, and appropriately trust the system.

---

## Requirements

```bash
pip install networkx numpy pandas matplotlib scikit-learn umap-learn seaborn
```

| Library | Purpose |
|---|---|
| NetworkX | Graph construction and network analysis |
| NumPy / Pandas | Numerical computation and data handling |
| scikit-learn | ML model, preprocessing, explainability |
| Matplotlib / Seaborn | Visualisation and dashboard outputs |

---

## Usage

Run the notebook cell by cell:

```bash
jupyter notebook flood_risk_hitl.ipynb
```

All outputs (three PNG figures) are saved automatically to the working directory.

---

## Connection to PhD Research

This project is built as preparation for doctoral research into Human-in-the-Loop AI for climate-resilient and equitable sewer systems (QUEX Institute, University of Exeter and University of Queensland). The specific connections are:

**Physics-informed GNNs:** The graph-based network representation and hydraulic constraint embedding in this project is a simplified precursor to full GNN surrogate digital twins that propagate hydraulic state across the pipe network using message passing between nodes.

**Scenario generation:** The stochastic climate scenario framework here uses parametric sampling. The PhD will extend this using diffusion model-based scenario generators that downscale CMIP6 global climate projections and integrate urban growth trajectories.

**Reinforcement learning with human feedback:** The decision dashboard in this project represents a static human-in-the-loop interface. The PhD will extend this to a dynamic RLHF framework where engineer feedback iteratively improves the AI's intervention recommendations.

**Equity considerations:** Future extensions of this work will incorporate spatial equity metrics — identifying whether high-risk nodes disproportionately serve marginalised communities, aligning with UN SDG 6 (clean water) and SDG 11 (sustainable cities).

---

## Next Steps

- Replace synthetic network with real OpenStreetMap drainage data using OSMnx
- Implement a full GNN surrogate using PyTorch Geometric for hydraulic state propagation
- Integrate CMIP6 climate projection downscaling for realistic future scenario generation
- Add an interactive RLHF interface where engineers provide feedback on AI-proposed interventions
- Validate against real sewer overflow data from Exeter or Brisbane case studies

---

## References

Gironás, J., Roesner, L.A., Rossman, L.A., & Davis, J. (2010). A new applications manual for the Storm Water Management Model (SWMM). *Environmental Modelling and Software*, 25(6), 813-814.

Zheng, J., Fu, G., et al. (2023). Physics-informed graph neural networks for sewer system modelling. *Water Research*.

Christiano, P., et al. (2017). Deep reinforcement learning from human preferences. *Advances in Neural Information Processing Systems*, 30.

QUEX Institute (2026). Explainable AI for Sustainable Sewers under Climate and Urban Stress. University of Exeter and University of Queensland Joint PhD Programme.

---

## Author

**Kunal Kamble**
MSc Advanced Computer Science, University of Liverpool
[LinkedIn](https://linkedin.com/in/kunal-kamble19) | [Email](mailto:kamblekunal165@gmail.com)

*Built in preparation for the QUEX Institute joint PhD programme between the University of Exeter and the University of Queensland, supervised by Dr. Jawad Fayaz.*
