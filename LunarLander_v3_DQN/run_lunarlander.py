import gymnasium as gym
import torch
import torch.nn as nn


# ==========================================
# MESMA ARQUITETURA USADA NO TREINAMENTO
# ==========================================

class DQN(nn.Module):

    def __init__(self, state_size, action_size):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(state_size, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, action_size)

        )

    def forward(self, x):

        return self.net(x)


# ==========================================
# CONFIGURAÇÕES
# ==========================================

STATE_SIZE = 8
ACTION_SIZE = 4

MODEL_FILE = "lunarlander_dqn.pth"


# ==========================================
# CARREGAR MODELO TREINADO
# ==========================================

model = DQN(
    STATE_SIZE,
    ACTION_SIZE
)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=torch.device("cpu")
    )
)

model.eval()

print("Modelo carregado com sucesso.")


# ==========================================
# CRIAR AMBIENTE COM ANIMAÇÃO
# ==========================================

env = gym.make(
    "LunarLander-v3",
    render_mode="human"
)

# ==========================================
# EXECUTAR AGENTE
# ==========================================

state, info = env.reset()

done = False

total_reward = 0

while not done:

    state_tensor = torch.FloatTensor(
        state
    ).unsqueeze(0)

    with torch.no_grad():

        q_values = model(
            state_tensor
        )

        action = torch.argmax(
            q_values
        ).item()

    state, reward, terminated, truncated, info = env.step(
        action
    )

    total_reward += reward

    done = terminated or truncated

env.close()

print(
    f"Reward total: {total_reward:.2f}"
)