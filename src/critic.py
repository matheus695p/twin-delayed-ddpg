import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    """
    Construimos una red neuronal para los criticos, se saca el número
    de neuronas del paper que Adressing functions aproximators que esta
    en la carpeta documents/

    ...

    Attributes
    ----------
    object : herencia
        módulo de redes neuronales de pytorch
    Methods
    -------
    forward(x):
        propagar adelante la red.
    Q1(x, u):
        propagar adelante la red.
    """

    def __init__(self, state_dim, action_dim):
        """
        Construcor de la redes
        Parameters
        ----------
        state_dim : int
            dimension de los espacios de estados.
        action_dim : int
            numero de acciones posibles.
        Returns
        -------
        None.

        """
        # herencia
        super(Critic, self).__init__()
        # Definimos el primero de los Críticos como red neuronal profunda
        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, 1)
        # Definimos el segundo de los Críticos como red neuronal profunda
        self.layer_4 = nn.Linear(state_dim + action_dim, 400)
        self.layer_5 = nn.Linear(400, 300)
        self.layer_6 = nn.Linear(300, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        # Propagación hacia adelante del primero de los Críticos
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        # Propagación hacia adelante del segundo de los Críticos
        x2 = F.relu(self.layer_4(xu))
        x2 = F.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1
