{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "grid_sarsa.py",
      "provenance": [],
      "collapsed_sections": [],
      "authorship_tag": "ABX9TyOZS2un23UKIVJK/ScwLCJM",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/shamim237/DenseNet50-Model-Implementation/blob/master/grid_sarsa.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "_cVA6dXVXgPk"
      },
      "source": [
        "from grid import Grid\n",
        "import numpy as np\n",
        "import random\n",
        "\n",
        "# Create environment\n",
        "env = Grid()\n",
        "\n",
        "episodes = 1000\n",
        "max_steps = 1000\n",
        "gamma = 0.9\n",
        "lr = 0.1\n",
        "decay = 0.8\n",
        "epsilon = 1\n",
        "epsilon_decay_rate = 0.003\n",
        "\n",
        "#Initializing the Q-matrix\n",
        "Q = np.zeros((len(env.stateSpace), len(env.actionSpace))) \n",
        "\n",
        "for ep in range(episodes):\n",
        "\n",
        "    env.reset()\n",
        "    state1 = env.currentState\n",
        "\n",
        "    mat = np.zeros((len(env.stateSpace), len(env.actionSpace)))\n",
        "\n",
        "    if random.uniform(0, 1) < epsilon:\n",
        "        action1 = random.choice(env.actionSpace)\n",
        "    else:\n",
        "        action1 = np.argmax(Q[env.currentState])\n",
        "        \n",
        "\n",
        "    for ms in range(max_steps):\n",
        "\n",
        "        state2, reward, done = env.step(action1)\n",
        "\n",
        "        if random.uniform(0, 1) < epsilon:\n",
        "            action2 = random.choice(env.actionSpace)\n",
        "        else:\n",
        "            action2 = np.argmax(Q[env.currentState])\n",
        "\n",
        "        td_error = reward + gamma * Q[state2, action2] - Q[state1, action1]\n",
        "\n",
        "        mat[state1, action1] += 1\n",
        "\n",
        "        for s in env.stateSpace:\n",
        "            for a in env.actionSpace:\n",
        "                Q[s, a] += lr * td_error * mat[s, a]\n",
        "                mat[s, a] = gamma * decay * mat[s, a]\n",
        "\n",
        "        state1 = state2\n",
        "        action1 = action2\n",
        "\n",
        "        if done:\n",
        "            break\n",
        "\n",
        "        epsilon = np.exp(-epsilon_decay_rate * ep)\n",
        "\n",
        "print('Action-Value function:')\n",
        "\n",
        "print (\"Last_State : \", env.currentState)\n",
        "print(Q)\n",
        "\n",
        "env.startGrid(Q)"
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}