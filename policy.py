import numpy as np
from rl.policy import Policy
import math

import logging
logger = logging.getLogger(__name__)
stream_log = logging.StreamHandler()
stream_log.setLevel(logging.DEBUG)
stream_log.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
logger.addHandler(stream_log)
logger.setLevel(logging.DEBUG)


class MaxBoltzmannQPolicy(Policy):
    """
    A combination of the eps-greedy and Boltzman q-policy.
    Wiering, M.: Explorations in Efficient Reinforcement Learning.
    PhD thesis, University of Amserdam, Amsterdam (1999)
    https://pure.uva.nl/ws/files/3153478/8461_UBA003000033.pdf
    """
    def __init__(self, eps=.1, tau=1., clip=(-500., 500.)):
        super(MaxBoltzmannQPolicy, self).__init__()
        self.eps = eps
        self.tau = tau
        self.clip = clip
        self.step = 0

    def select_action(self, q_values):
        assert q_values.ndim == 1
        self.tau = 1/math.log(self.step + 1.1)
        logger.debug(q_values)
        q_values = q_values.astype('float64')
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
            probs = exp_values / np.sum(exp_values)
            action = np.random.choice(range(nb_actions), p=probs)
        else:
            action = np.argmax(q_values)
        self.step += 1
        return action

    def get_config(self):
        config = super(MaxBoltzmannQPolicy, self).get_config()
        config['eps'] = self.eps
        config['tau'] = self.tau
        config['clip'] = self.clip
