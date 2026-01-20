## A2C (Advantage Actor-Critic) Implementation with PyTorch
"Reducing Variance in Policy Gradient Methods"

This project implements the A2C (Advantage Actor-Critic) algorithm to solve the CartPole-v1 environment.

It focuses on improving the stability and data efficiency of the vanilla REINFORCE algorithm by introducing a Critic network and the concept of Advantage.

# Core Concept: Why A2C?
The primary motivation for A2C is to solve the High Variance problem of the REINFORCE algorithm.
  
  - Variance Reduction (REINFORCE vs A2C)REINFORCE (Monte-Carlo): Updates the policy based on the total return 'G_t' of a single episode.
  - Problem: The return varies wildly depending on 'luck', leading to unstable learning (High Variance).
  - A2C (TD Learning): Updates using the Advantage function.
  - Solution: It introduces a Critic (V(s)) to serve as a baseline. It evaluates actions not by absolute score, but by how much better they are compared to the                 average expectation. A(s, a) = r + gamma * V(s') - V(s)
  - Visual Comparison: As shown in the graph below, A2C shows a smoother and more stable learning curve compared to the fluctuating performance of REINFORCE.
   ![Learning Curve Comparison]([./a2c_score.png][./reinforce_score.png])



## Model Architecture: Shared Network
This implementation uses a Shared Network Architecture where both the Actor and Critic share the lower layers (Feature Extractor) but have separate output heads.
ActorCritic(nn.Module):
    def __init__(self):
        # Shared Body (Feature Extractor)
        self.fc1 = nn.Linear(4, 256)
        # Head 1: Actor (Policy) -> Probabilities
        self.fc_actor = nn.Linear(256, 2)
        # Head 2: Critic (Value) -> Scalar Score
        self.fc_critic = nn.Linear(256, 1)
        
## Challenges in Shared Architecture (Analysis)
While sharing layers is computationally efficient and helps with feature extraction, it introduces two critical challenges that I analyzed during implementation:

### Challenge 1: Scale Imbalance (체급 차이)
- Issue: The magnitude of the Critic Loss (MSE) is often much larger than the Actor Loss (Log Probability).
- Consequence: The optimizer may prioritize minimizing the Critic Loss, causing the Actor (Policy) to learn too slowly or stagnate.
- Solution: In complex environments, we could apply a coefficient to balance the scales (e.g., L = L_actor + 0.5 * L_critic).

### Challenge 2: Gradient Interference (줄다리기 문제)
- Issue: The Actor and Critic optimize for different objectives.
Actor: Wants to shift parameters to increase probability of good actions.
Critic: Wants to shift parameters to predict value accurately.
- Consequence: The gradients may conflict (Destructive Interference) in the shared layers, canceling each other out and slowing down convergence.
- Trade-off: Despite these risks, I used a shared architecture for CartPole-v1 as the benefits of shared feature learning outweigh the interference in simple environments.

## How to Run
Open Actor-Critic.ipynb in Google Colab
