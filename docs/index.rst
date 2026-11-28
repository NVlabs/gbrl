.. GBRL documentation master file, created by
   sphinx-quickstart on Mon Jun  3 06:40:46 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GBRL's documentation!
================================
GBRL is a Python-based Gradient Boosting Trees (GBT) library, similar to popular packages such as `XGBoost <https://xgboost.readthedocs.io/en/stable/>`__ , `CatBoost <https://catboost.ai/>`__ , but specifically designed and optimized for reinforcement learning (RL). GBRL is implemented in C++/CUDA aimed to seamlessly integrate within popular RL libraries. 

GBRL also provides the core tree-learning implementation of
`Split-RL: Local Conflict Resolution in Reinforcement Learning <https://arxiv.org/abs/2611.34799>`__
(NeurIPS 2026): multi-objective gradients, conflict-aware tree splitting, and
reward/cost critics. See :doc:`split_rl` for usage and
`NVlabs/split_rl <https://github.com/NVlabs/split_rl>`__ for the algorithm and
experiment code.

Feature Support Matrix
----------------------

The following table summarizes feature availability by tree type and device:

+----------------------------+---------------------+-------------------+---------------------+-------------------+
| Feature                    | Greedy CPU          | Greedy GPU        | Oblivious CPU       | Oblivious GPU     |
+============================+=====================+===================+=====================+===================+
| Tree Fitting               | ✓                   | ✓                 | ✓                   | ✓                 |
+----------------------------+---------------------+-------------------+---------------------+-------------------+
| Monotonic Constraints      | ✗                   | ✗                 | ✓ (SGD only)        | ✓ (SGD only)      |
+----------------------------+---------------------+-------------------+---------------------+-------------------+
| Linear LR Scheduler        | ✓                   | ✗                 | ✓                   | ✓                 |
+----------------------------+---------------------+-------------------+---------------------+-------------------+
| Constant LR Scheduler      | ✓                   | ✓                 | ✓                   | ✓                 |
+----------------------------+---------------------+-------------------+---------------------+-------------------+
| ADAM Optimizer             | ✓                   | ✗                 | ✓                   | ✗                 |
+----------------------------+---------------------+-------------------+---------------------+-------------------+
| SGD Optimizer              | ✓                   | ✓                 | ✓                   | ✓                 |
+----------------------------+---------------------+-------------------+---------------------+-------------------+
| Control Variates           | ✓                   | ✗                 | ✓                   | ✗                 |
+----------------------------+---------------------+-------------------+---------------------+-------------------+
| Split-RL (multi-objective) | ✓                   | ✓                 | ✓                   | ✓                 |
+----------------------------+---------------------+-------------------+---------------------+-------------------+

.. note::

   Monotonic constraints apply to the output dimensions covered by the optimizer
   (``start_idx`` to ``stop_idx-1``). They require **SGD**: GBRL raises a ``ValueError``
   if *any* optimizer on the model uses Adam, and ``MultiGBTLearner`` does not support
   them at all. See :doc:`examples` for details.

.. note::

   Split-RL models (``n_objs > 1``) train through ``step``; ``fit`` and control
   variates are not supported with multiple objectives. See :doc:`split_rl`.

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   quickstart
   examples
   split_rl

.. toctree::
   :maxdepth: 2
   :caption: Documentation:
   :hidden:

   models/index
   learners/index

Citation
--------

For GBRL:

.. code-block:: bibtex

   @inproceedings{
   fuhrer2025gradient,
   title={Gradient Boosting Reinforcement Learning},
   author={Benjamin Fuhrer and Chen Tessler and Gal Dalal},
   booktitle={Forty-second International Conference on Machine Learning},
   year={2025},
   url={https://arxiv.org/abs/2407.08250}
   }

For work using the Split-RL extensions, please also cite:

.. code-block:: bibtex

   @inproceedings{
   fuhrer2026splitrl,
   title={Split-{RL}: Local Conflict Resolution in Reinforcement Learning},
   author={Benjamin Fuhrer and Chen Tessler and Gal Dalal},
   booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
   year={2026},
   url={https://arxiv.org/abs/2611.34799}
   }
