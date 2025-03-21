SeparateActorCriticLearner
===========================

`SeparateActorCriticLearner` uses two distinct gradient boosted tree learners to represent an actor and a critic (Value function).
Useful when the actor and critic need to be trained independently with different tree configurations or update rules.
It is a wrapper around `MultiGBTLearner` and supports training, prediction, saving/loading, and SHAP value computation per ensemble.

.. autoclass:: gbrl.learners.actor_critic_learner.SeparateActorCriticLearner
   :members:
   :undoc-members:
   :show-inheritance:
