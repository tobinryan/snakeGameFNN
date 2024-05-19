# Python Snake Game Trained by a Feedforward Neural Network

This Python Snake Game is powered by a feedforward neural network (FNN). The game uses reinforcement learning techniques to train an AI agent to control the snake, allowing it to navigate the game board, collect apples, and avoid collisions with the wall and itself.

## Getting Started

To run the game and train the AI agent, follow these steps:

1. Ensure you have Python 3.x installed on your system.

2. Clone the repository or download the source code to your local machine.

3. Open a terminal or command prompt and navigate to the project directory.

4. Run the game by executing the following command:

   ```bash
   python trainer.py

The game will start, and the NN will begin training. You can watch the game's progress as it plays and learns.
## Game Controls
The AI agent controls the snake automatically. There's no need for manual controls during training. You can observe the neural network's performance and the game's progress.

## Customizing Game Speed
You can modify the speed of the game by adjusting the SPEED variable in the snake_game.py file. The SPEED variable determines the game's speed, allowing you to slow down or speed up the AI's training.

  ```bash
  # snake_game.py

  SPEED = 40  # Adjust this value to change the game speed
  ```
Simply change the value of SPEED to your preferred speed (in milliseconds per frame).

## Training and Observing
The game will display the current game number, the snake's score, and the best score achieved. You can monitor the training progress by watching the game and observing how the AI agent's performance improves over time.

## Game Features
The snake moves autonomously, controlled by the AI agent.
The AI agent learns to navigate the game board and collect apples.
You can customize the game speed to adjust the training pace.
The game visualizes the snake's performance and learning progress.

## Acknowledgments
This Python Snake Game with a feedforward neural network was created as a fun and educational project. It demonstrates how reinforcement learning can be used to train AI agents in simple games. Feel free to explore and experiment with the code to learn more about the training process and AI-controlled games.

Enjoy playing and observing the network's training journey in this Snake Game!
