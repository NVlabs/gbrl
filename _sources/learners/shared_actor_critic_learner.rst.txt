SharedActorCriticLearner
========================

`SharedActorCriticLearner` uses a single gradient boosted tree learners to represent both an actor and a critic (Value function).
Useful when the actor and critic need to share tree configurations or update rules.
It is a wrapper around `GBTLearner` and supports training, prediction, saving/loading, and SHAP value computation per ensemble.

.. autoclass:: gbrl.learners.actor_critic_learner.SharedActorCriticLearner
   :members:
   :undoc-members:
   :show-inheritance:
