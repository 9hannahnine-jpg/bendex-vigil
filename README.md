# Arc Vigil

Detect. Attribute. Intervene.

Real-time neural network training stability monitoring grounded in information geometry.

## What it does

Arc Vigil monitors your training run in real time, identifies which layer is failing, and corrects it automatically — before your loss curve shows anything.

| | Arc Vigil | Loss spike | Gradient norm | Patience |
|--|--|--|--|--|
| Detection rate | 100% | 100% | 90% | 100% |
| False positive rate | 0% | 80% | 50% | 50% |
| Recovery rate | 90% | 0% | 0% | 0% |
| Attributes module | Yes | No | No | No |
| Intervenes automatically | Yes | No | No | No |

30-seed benchmark across 6 architectures: MLP, CNN, ViT, DistilBERT, GPT-2, ResNet-50 + LR spike stress test.

## How it works

Arc Vigil treats your model's weight trajectory as a geometric object and computes discrete curvature over it:

    k_t = |d_t - 2*d_{t-1} + d_{t-2}|

where d_t is the normalized per-module weight divergence from a reference checkpoint. During healthy training, k_t stays near zero. At instability onset it spikes — well before loss diverges.

Two detection channels run in parallel:

- AND mode — divergence step-change AND gradient energy both persistently exceed their z-scored baseline
- Kappa mode — curvature alone persistently exceeds its z-scored baseline

When a trigger fires, the deepest-early-cluster attribution algorithm identifies the specific module that deviated first. A three-phase intervention fires automatically.

## Install

    pip install arc-vigil

Or from source:

    git clone https://github.com/9hannahnine-jpg/arc-vigil
    cd arc-vigil
    pip install -e .

## Usage

    from arc_vigil import BendexMonitor, BendexConfig, BendexIntervention

    monitor = BendexMonitor(model, config=BendexConfig())
    intervention = BendexIntervention(model, optimizer)

    for step, batch in enumerate(dataloader):
        event = monitor.observe(step)
        intervention.step(event, step)

        loss = model(batch)
        loss = loss + intervention.logit_loss(logits)
        loss.backward()
        intervention.apply_grad_clip()
        optimizer.step()
        optimizer.zero_grad()

## Configuration

    config = BendexConfig(
        warmup_steps=60,   # steps to collect baseline before detection starts
        z_threshold=2.5,   # z-score threshold for detection
        persist=2,         # consecutive steps signal must exceed threshold
        max_lag=6,         # attribution temporal window
    )

## Validated architectures

- Multilayer perceptron (MLP)
- Convolutional neural network (CNN)
- Small Vision Transformer (ViT)
- DistilBERT 66M
- GPT-2 117M
- ResNet-50

## Also available

- Arc Sentry — prompt injection detection for open source LLMs. pip install arc-sentry
- Arc Gate — behavioral monitoring proxy for closed model APIs (GPT-4, Claude, Gemini). bendexgeometry.com/gate

## Research

Arc Vigil is the applied proof of a theoretical program in information geometry. Five published papers:

1. Informational Curvature
2. Informational Stability
3. Reflexivity
4. Experience
5. Specialization — derives the fine structure constant to 8 significant figures from pure geometry

Full theory: bendexgeometry.com/theory

## License

Source available. Free for research and non-commercial use. Commercial license required for all other use.

Patent pending. Methods covered by provisional patent applications filed by Hannah Nine / Bendex Geometry LLC (priority dates November 2025, February 2026, March 2026).

For commercial licensing: 9hannahnine@gmail.com

## Author

Hannah Nine — bendexgeometry.com · ORCID 0009-0006-3884-7372
