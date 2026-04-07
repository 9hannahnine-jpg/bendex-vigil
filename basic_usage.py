"""
Basic Vigil usage example.
Drop observe() into your training loop before the forward pass.
That's all it takes.
"""
import torch
import torch.nn as nn
from bendex import BendexMonitor, BendexConfig, BendexIntervention

# --- Your model and optimizer ---
model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10),
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- Set up Vigil ---
config = BendexConfig(
    warmup_steps=60,
    z_threshold=2.5,
    persist=2,
)
monitor = BendexMonitor(model, config=config)
intervention = BendexIntervention(model, optimizer)

# --- Training loop ---
for step in range(500):
    # 1. Observe BEFORE forward pass
    event = monitor.observe(step)
    # 2. Fire intervention if event detected
    intervention.step(event, step)
    if event:
        print(f"Step {step}: instability detected in '{event['suspect_module']}' "
              f"via {event['mode']} mode (z_kappa={event['z_kappa']:.2f})")
    # 3. Forward pass
    x = torch.randn(32, 128)
    y = torch.randint(0, 10, (32,))
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, y)
    # 4. Optional: add logit penalty after intervention
    loss = loss + intervention.logit_loss(logits)
    # 5. Backward
    loss.backward()
    # 6. Optional: apply grad clip after intervention
    intervention.apply_grad_clip()
    optimizer.step()
    optimizer.zero_grad()

print(f"\nTotal events detected: {len(monitor.events)}")
for e in monitor.events:
    print(f"  Step {e['step']}: {e['suspect_module']} ({e['mode']} mode)")
