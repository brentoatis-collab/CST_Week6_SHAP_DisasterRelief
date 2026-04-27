# CST 643 Week 6 — Vibe Coding Log  
## From Implementation to Investigation

This project demonstrates that building an RL agent is only half the problem — validating and explaining its behavior is what makes it deployable in real-world, high-stakes systems.

---

## Project Objective

The goal of this project was to design and train a reinforcement learning (RL) agent in a GridWorld disaster relief scenario, and more importantly, to interpret the agent’s decision-making using SHAP.

This week emphasized a critical shift:
From building an agent → to understanding an agent.

---

## Phase 1 — Environment & Agent Implementation

Prompt:
"I am building a reinforcement learning disaster relief agent using GridWorld. Help me design an environment with obstacles, terrain rewards, and priority zones."

Outcome:
- Built a 5x5 GridWorld environment
- Defined terrain types: Critical, Moderate, Low, Delay, Obstacle, Goal
- Implemented reward structure aligned with disaster priorities

---

## Phase 2 — Q-Learning Agent Development

Prompt:
"Help me implement a Q-learning agent with epsilon-greedy exploration in a GridWorld environment."

Outcome:
- Implemented Q-table using defaultdict
- Applied epsilon-greedy strategy for exploration vs exploitation
- Trained agent over multiple episodes
- Generated outputs:
  - Reward trend plot
  - Learned policy grid
  - Policy heatmap

Insight:
The agent successfully learned to prioritize critical zones and avoid obstacles, demonstrating effective reward-based learning.

---

## Phase 3 — SHAP Interpretability Pipeline

Prompt:
"I have a trained RL agent. Help me apply SHAP to understand why the agent selects certain actions in different states."

Outcome:
- Engineered interpretable features:
  - Position (x, y)
  - Distance to critical zones
  - Distance to obstacles
- Built SHAP dataset from agent trajectories
- Created SHAP analysis pipeline using KernelExplainer
- Generated SHAP summary plot

Key Learning:
SHAP provides feature-level explanations, allowing visibility into the agent’s decision-making logic.

---

## Phase 4 — Debugging & Investigation (Critical Turning Point)

Prompt:
"My SHAP analysis appears to hang or never complete. Help diagnose where the issue might be occurring."

Investigation Process:
- Initially suspected SHAP computation complexity
- Reduced dataset size and sampling parameters
- Added debug print statements to isolate execution stage

Root Cause Identified:
The issue was not SHAP itself, but the dataset generation process.

The RL agent occasionally entered **infinite loops**, failing to reach a terminal state (`done=True`).  
This caused the `while not done` loop in `build_shap_dataset()` to run indefinitely.

---

## Resolution

Implemented a safeguard:

- Introduced `max_steps` constraint within dataset generation loop
- Modified loop condition:
  `while not done and steps < max_steps`

Outcome:
- Prevented infinite loops
- Stabilized dataset generation
- Enabled successful SHAP execution

---

## Phase 5 — SHAP Interpretation & Insights

Prompt:
"Help me interpret SHAP results and explain what they reveal about the agent’s behavior."

Key Findings:
- Distance to critical zones was the most influential feature
- Obstacle proximity became significant near hazard areas
- Position contributed minor directional influence

Interpretation:
The agent demonstrated **context-aware decision-making**:
- Prioritizing high-need areas
- Adjusting behavior near obstacles for safety

This confirmed alignment with both:
- Operational objectives (efficiency)
- Ethical objectives (fair resource distribution)

This confirms the agent is not simply maximizing reward blindly, but is learning a structured decision policy that balances urgency (critical zones) with risk management (obstacles), which is essential in disaster response scenarios.

In real-world applications such as disaster relief, healthcare triage, or autonomous navigation, this level of interpretability is critical to ensure that decisions are not only efficient but also justifiable and fair.

---

## Vibe Coding Reflection

This project demonstrated that vibe coding is not just an implementation tool, but an **investigative framework**.

Claude Code was used to:
- Design the RL environment
- Build the Q-learning agent
- Construct the SHAP analysis pipeline
- Diagnose and debug system-level issues
- Interpret model behavior

Key Insight:
Understanding an AI system requires more than building it —  
it requires structured analysis, debugging discipline, and interpretability tools.

Vibe coding transformed this workflow from code generation into iterative analysis, where each prompt served as a hypothesis, and each output became evidence for validating the agent’s behavior.

---

## Final Takeaway

Interpretability is not automatic — it must be engineered.

By combining reinforcement learning with SHAP and a structured investigative workflow, this project transformed a black-box agent into a transparent and explainable decision-making system.