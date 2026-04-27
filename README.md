# CST 643 Week 6  
## Understanding Decision-Making in Reinforcement Learning Using SHAP

---

## Project Overview

This project explores how reinforcement learning (RL) agents make decisions in a disaster relief scenario and, more importantly, how those decisions can be interpreted using SHAP (SHapley Additive Explanations).

The objective is not only to build an effective RL agent, but to transform a traditionally “black box” system into a transparent, explainable decision-making model.

This project demonstrates that building an RL agent is only half the problem — validating and explaining its behavior is what makes it deployable in real-world, high-stakes systems.

---

## Problem Statement

In disaster relief operations, resources such as food, water, and medical supplies must be distributed efficiently and fairly.  

The RL agent must:
- Navigate a grid-based disaster environment  
- Avoid obstacles and delays  
- Prioritize critical zones  
- Balance efficiency with fairness  

The challenge lies in understanding *why* the agent makes its decisions—not just what it does.

---

## GridWorld Environment

The environment is a 5x5 grid representing a disaster-affected region.

### Terrain Types:
- **C (Critical)** → High priority / high reward  
- **M (Moderate)** → Medium priority  
- **L (Low)** → Low priority  
- **D (Delay)** → Movement penalty  
- **X (Obstacle)** → Impassable  
- **G (Goal)** → Terminal state  
- **S (Start)** → Agent starting point  

### Objective:
Maximize reward by efficiently reaching high-priority areas while avoiding penalties.

---

## Reinforcement Learning Agent

The agent is trained using **Q-learning** with an **epsilon-greedy strategy**.

### Key Components:
- Q-table initialized using `defaultdict`
- Exploration vs exploitation balancing
- Reward-based learning over multiple episodes

### Evaluation Metrics:
- Average reward over episodes  
- Steps per episode  
- Learned policy grid  
- Policy heatmap visualization  

### Key Outcome:
The agent successfully learned to:
- Prioritize critical zones  
- Avoid obstacles  
- Develop efficient navigation paths  

---

## SHAP Interpretability

SHAP, or Shapley Additive Explanations, allows us to interpret model decisions by assigning importance values to each feature.

SHAP is based on game-theoretic Shapley values, which provide a fair and consistent way to attribute each feature’s contribution to a prediction. Unlike simpler methods that rely on static weights or correlations, SHAP evaluates feature impact across multiple combinations, making it more robust and reliable for interpreting complex RL behavior.

### Features Used:
- Position (x, y)  
- Distance to critical zones  
- Distance to obstacles  

### SHAP Outputs:
- Summary plot showing global feature importance  
- Feature contribution analysis  

---

## SHAP Insights

Key findings from SHAP analysis:

- **Distance to critical zones** is the most influential feature  
- **Obstacle proximity** becomes important near hazards  
- Position has minor directional influence  

### Interpretation:
The agent demonstrates **context-aware decision-making**:
- Prioritizing urgent needs (critical zones)  
- Adjusting behavior near obstacles for safety  

This confirms the agent is not simply maximizing reward blindly, but is learning a structured decision policy that balances urgency and risk management—critical for disaster response scenarios.

In real-world applications such as disaster relief, healthcare triage, or autonomous navigation, this level of interpretability is essential to ensure decisions are efficient, fair, and justifiable.

This validates that the agent’s learned policy is not only optimal in terms of reward, but also aligned with real-world decision priorities, where urgency and safety must be balanced simultaneously.

---

## Debugging & Investigation

A critical issue was encountered during SHAP dataset generation:

### Problem:
The agent entered an **infinite loop**, preventing SHAP from completing.

### Root Cause:
The loop:
```python
while not done: did not always terminate because the agent failed to reach a terminal state.
```

### Solution:

To prevent infinite loops, a safeguard was introduced:

```python
max_steps = 50
steps = 0

while not done and steps < max_steps:
    ...
    steps += 1
```

### Outcome:

- Eliminated infinite loop risk  
- Stabilized dataset generation  
- Enabled SHAP analysis to complete successfully  

---

## Vibe Coding Approach

This project reflects a fundamental shift in workflow:

**From building an RL agent → to understanding an RL agent**

Instead of using AI assistance purely for implementation, it was used as an **investigation tool**.

Claude Code supported:

- Environment design and structure  
- Q-learning agent implementation  
- SHAP pipeline construction  
- Debugging and root cause analysis  
- Interpretation of model behavior  

### Key Insight:

Vibe coding transformed development into an **iterative analytical process**, where:

- Each prompt acted as a hypothesis  
- Each output served as validation or contradiction  
- Debugging became structured investigation 

---

## Project Structure

CST_Week6_SHAP_DisasterRelief/
│
├── main.py
├── README.md
├── requirements.txt
├── claude_log.md
│
├── outputs/
│   ├── training_rewards.png
│   ├── policy_value_heatmap.png
│   ├── shap_summary.png
│
└── slides/
    └── Week6_Final_Presentation.pptx

---

## How to Run the Project

### 1. Activate virtual environment
```bash
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirement.txt
```

### 3. Run the project
```bash
python3 main.py
```

---

## Outputs

The project generates the following outputs:

- Training reward plot (training_rewards.png)
- Policy value heatmap (policy_value_heatmap.png)
- SHAP summary plot (shap_summary.png)

The learned policy is displayed in the terminal as a directional grid visualization.

All outputs are stored in the /outputs directory.

All visual outputs (training rewards, policy heatmap, SHAP plots) are generated automatically when running the project.

---

## References

GeeksforGeeks. (2024, January 3). SHAP: A comprehensive guide to SHAPley additive explanations.
https://www.geeksforgeeks.org/shap-a-comprehensive-guide-to-shapley-additive-explanations/

GeeksforGeeks. (2025, February 25). Q-learning in reinforcement learning.
https://www.geeksforgeeks.org/q-learning-in-python/

---

## Final Takeaway

Reinforcement learning models are powerful, but without interpretability, they remain incomplete.

This project demonstrates that true understanding comes not from observing outcomes, but from analyzing the reasoning behind them.

By combining RL with SHAP and a structured investigative workflow, this system evolves from a black box into a transparent, accountable decision-making framework suitable for real-world deployment.