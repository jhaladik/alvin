"""
Reinforcement Learning Training Script
Hybrid Approach: Bootstrap from human demonstrations, then improve via self-play
AI ACTUALLY PLAYS THE GAME AND LEARNS!
"""
import torch
import numpy as np
import sys
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.pacman_env import PacManEnv
from agents.dqn_agent import DQNAgent


def train_rl(config):
    """
    Main RL training loop

    The AI plays the game repeatedly and learns from experience!
    """
    print("=" * 70)
    print("Alvin Pac-Man - REINFORCEMENT LEARNING Training")
    print("=" * 70)
    print("\nHybrid Approach:")
    print("  Phase 1: Bootstrap from human demonstrations")
    print("  Phase 2: Improve via self-play (RL)")
    print()

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Create environment
    print("[1/3] Creating Pac-Man environment...")
    env = PacManEnv(
        maze_size=config['maze_size'],
        max_steps=config['max_steps_per_episode']
    )
    print(f"  Maze: {config['maze_size']}")
    print(f"  Max steps per episode: {config['max_steps_per_episode']}")

    # Create agent
    print("\n[2/3] Creating DQN agent...")
    agent = DQNAgent(
        state_dim=config['state_dim'],
        action_dim=config['action_dim'],
        hidden_dims=config['hidden_dims'],
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        epsilon_start=config['epsilon_start'],
        epsilon_end=config['epsilon_end'],
        epsilon_decay=config['epsilon_decay'],
        buffer_capacity=config['buffer_capacity'],
        device=device,
        pretrained_path=config.get('pretrained_path')
    )
    print(f"  Parameters: {sum(p.numel() for p in agent.q_network.parameters()):,}")
    print(f"  Initial epsilon: {agent.epsilon:.3f}")

    # TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/rl_{timestamp}"
    writer = SummaryWriter(log_dir)
    print(f"\n  TensorBoard: {log_dir}")

    # Training loop
    print("\n[3/3] Starting RL training...")
    print("  The AI will now play Pac-Man and learn from experience!\n")
    print("=" * 70)

    best_avg_reward = -float('inf')
    episode_rewards = []

    for episode in range(config['num_episodes']):
        # Reset environment
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_loss = 0
        done = False

        while not done:
            # AI selects action (epsilon-greedy)
            action = agent.select_action(state, training=True)

            # Execute action in environment
            next_state, reward, done, info = env.step(action)

            # Store transition in replay buffer
            agent.store_transition(state, action, reward, next_state, done)

            # Train on random batch
            if len(agent.replay_buffer) > config['min_buffer_size']:
                loss = agent.train_step(batch_size=config['batch_size'])
                episode_loss += loss

            episode_reward += reward
            episode_steps += 1
            state = next_state

        # Update target network periodically
        if (episode + 1) % config['target_update_freq'] == 0:
            agent.update_target_network()

        # Decay epsilon
        agent.decay_epsilon()

        # Track statistics
        episode_rewards.append(episode_reward)
        avg_reward_100 = np.mean(episode_rewards[-100:])

        # Log to TensorBoard
        writer.add_scalar('Reward/episode', episode_reward, episode)
        writer.add_scalar('Reward/avg_100', avg_reward_100, episode)
        writer.add_scalar('Loss/episode', episode_loss / max(episode_steps, 1), episode)
        writer.add_scalar('Epsilon', agent.epsilon, episode)
        writer.add_scalar('Steps/episode', episode_steps, episode)
        writer.add_scalar('Score/episode', info['score'], episode)
        writer.add_scalar('Lives/remaining', info['lives'], episode)

        # Print progress
        if (episode + 1) % config['log_freq'] == 0:
            print(f"Episode {episode+1}/{config['num_episodes']}")
            print(f"  Reward: {episode_reward:.1f}  Avg(100): {avg_reward_100:.1f}")
            print(f"  Score: {info['score']}  Lives: {info['lives']}  Steps: {episode_steps}")
            print(f"  Epsilon: {agent.epsilon:.3f}  Buffer: {len(agent.replay_buffer)}")
            print(f"  Loss: {episode_loss / max(episode_steps, 1):.4f}")
            print()

        # Save best model
        if avg_reward_100 > best_avg_reward and episode >= 100:
            best_avg_reward = avg_reward_100
            save_path = os.path.join(config['checkpoint_dir'], 'best_rl_model.pth')
            agent.save(save_path)
            print(f"  [+] New best model saved! Avg reward: {avg_reward_100:.1f}\n")

        # Save checkpoint periodically
        if (episode + 1) % config['save_freq'] == 0:
            save_path = os.path.join(config['checkpoint_dir'], f'rl_checkpoint_ep{episode+1}.pth')
            agent.save(save_path)

    # Final evaluation
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nFinal Statistics:")
    print(f"  Total episodes: {config['num_episodes']}")
    print(f"  Best avg reward (100 eps): {best_avg_reward:.1f}")
    print(f"  Final epsilon: {agent.epsilon:.3f}")
    print(f"  Total transitions in buffer: {len(agent.replay_buffer)}")

    # Evaluate trained agent
    print("\nEvaluating trained agent (10 episodes)...")
    eval_rewards = []
    eval_scores = []

    for i in range(10):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, training=False)  # No exploration
            state, reward, done, info = env.step(action)
            episode_reward += reward

        eval_rewards.append(episode_reward)
        eval_scores.append(info['score'])
        print(f"  Eval {i+1}: Reward={episode_reward:.1f}, Score={info['score']}")

    print(f"\nEvaluation Results:")
    print(f"  Avg Reward: {np.mean(eval_rewards):.1f}")
    print(f"  Avg Score: {np.mean(eval_scores):.1f}")

    writer.close()

    print("\n[+] RL Training complete!")
    print(f"  Best model: checkpoints/best_rl_model.pth")
    print(f"  TensorBoard: {log_dir}")

    return agent


def main():
    """Main entry point"""
    # Training configuration
    config = {
        # Environment
        'maze_size': (20, 20),
        'max_steps_per_episode': 1000,

        # Network architecture
        'state_dim': 128,
        'action_dim': 4,
        'hidden_dims': [256, 256, 128],

        # Training hyperparameters
        'num_episodes': 5000,  # AI will play 5000 games!
        'learning_rate': 0.0001,
        'gamma': 0.99,
        'batch_size': 64,
        'buffer_capacity': 100000,
        'min_buffer_size': 1000,

        # Exploration
        'epsilon_start': 1.0,    # Start with 100% exploration
        'epsilon_end': 0.01,     # End with 1% exploration
        'epsilon_decay': 0.995,  # Decay rate per episode

        # Updates
        'target_update_freq': 10,  # Update target network every N episodes

        # Logging & Saving
        'log_freq': 50,
        'save_freq': 500,
        'checkpoint_dir': 'checkpoints',

        # HYBRID: Load pre-trained model to bootstrap
        # Set to None to start from scratch
        'pretrained_path': 'checkpoints/best_dqn_model.pth'  # From behavioral cloning
    }

    # Create checkpoint directory
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    # Train
    agent = train_rl(config)

    print("\nNext steps:")
    print("  1. Check TensorBoard for training curves: tensorboard --logdir=runs")
    print("  2. Test the RL-trained model in inference_server.py")
    print("  3. Compare performance: Behavioral Cloning vs RL vs Heuristics")

    return 0


if __name__ == "__main__":
    exit(main())
