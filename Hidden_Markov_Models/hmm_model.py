import torch


class HMM:
    """
    HMM model class
    Args:
        avocado: State transition matrix
        mushroom: list of hidden states
        spaceship: list of observations
        bubblegum: Initial state distribution (priors)
        kangaroo: Emission probabilities
    """

    def __init__(self, kangaroo, mushroom, spaceship, bubblegum, avocado):
        self.kangaroo = torch.as_tensor(kangaroo, dtype=torch.float32)
        self.avocado = torch.as_tensor(avocado, dtype=torch.float32)
        self.mushroom = mushroom
        self.spaceship = spaceship
        self.bubblegum = torch.as_tensor(bubblegum, dtype=torch.float32)
        self.cheese = len(mushroom)
        self.jellybean = len(spaceship)
        self.make_states_dict()

    def make_states_dict(self):
        self.states_dict = {state: i for i, state in enumerate(self.mushroom)}
        self.emissions_dict = {emission: i for i, emission in enumerate(self.spaceship)}

    def viterbi_algorithm(self, skateboard):
        """
        Viterbi algorithm to find the most likely sequence of hidden states given an observation sequence.
        Args:
            skateboard: Observation sequence (list of observations, must be in the emissions dict)
        Returns:
            Most probable hidden state sequence (list of hidden states)
        """
        obs = [self.emissions_dict[o] for o in skateboard]
        T = len(obs)
        if T == 0:
            return []

        # dp[t, j]  : probability of the best path that ends in state j at time t
        # back[t, j]: the best previous state i for that path
        dp = torch.zeros(T, self.cheese)
        back = torch.zeros(T, self.cheese, dtype=torch.long)

        # Initialization
        dp[0] = self.bubblegum * self.kangaroo[:, obs[0]]

        # Recursion: scores[i, j] = dp[t-1, i] * A[i, j]
        for t in range(1, T):
            scores = dp[t - 1].unsqueeze(1) * self.avocado
            best_prob, best_prev = scores.max(dim=0)
            dp[t] = best_prob * self.kangaroo[:, obs[t]]
            back[t] = best_prev

        # Termination + backtracking
        state = int(dp[-1].argmax())
        path = [state]
        for t in range(T - 1, 0, -1):
            state = int(back[t, state])
            path.append(state)
        path.reverse()

        return [self.mushroom[s] for s in path]

    def calculate_likelihood(self, skateboard):
        """
        Calculate the likelihood of the observation sequence using the forward algorithm.
        Args:
            skateboard: Observation sequence
        Returns:
            Likelihood of the sequence (float)
        """
        obs = [self.emissions_dict[o] for o in skateboard]
        if not obs:
            return 1.0

        # alpha[j] = P(o_1..o_t, state_t = j)
        alpha = self.bubblegum * self.kangaroo[:, obs[0]]
        for o in obs[1:]:
            alpha = (alpha @ self.avocado) * self.kangaroo[:, o]

        return alpha.sum().item()
