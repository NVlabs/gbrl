Models
======

This section documents high-level models that wrap one or more learners to expose a complete API for training, evaluation, and gradient boosting operations in reinforcement learning and supervised settings.

Each model inherits from a common base interface (`BaseGBT`) and typically contains logic for managing boosting steps, SHAP explanations, device movement, and gradient tracking.

.. toctree::
   :maxdepth: 1
   :caption: Classes:

   base_gbt
   gbt
   actor_critic
   continuous_critic
   discrete_critic
   gaussian_actor
   parametric_actor
