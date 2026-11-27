ActorCritic
===========

GBRL model implementing actor-critic architecture using gradient-boosted decision trees. Supports both shared and separate ensembles for the actor and critic, enabling flexible RL algorithm design such as PPO or A2C with tree-based models.

.. autoclass:: gbrl.models.actor_critic.ActorCritic
   :members:
   :undoc-members:
   :show-inheritance:

CostActorCritic
---------------

``CostActorCritic`` adds a scalar cost critic to the policy and reward critic.
Its outputs are ``(policy, reward_value, cost_value)``. See :doc:`../split_rl`
for objective-gradient conventions and current limitations.

.. autoclass:: gbrl.models.actor_critic.CostActorCritic
   :members:
   :show-inheritance:
