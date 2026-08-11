.. GBRL documentation master file, created by
   sphinx-quickstart on Mon Jun  3 06:40:46 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GBRL's documentation!
================================
GBRL is a Python-based Gradient Boosting Trees (GBT) library, similar to popular packages such as `XGBoost <https://xgboost.readthedocs.io/en/stable/>`__ , `CatBoost <https://catboost.ai/>`__ , but specifically designed and optimized for reinforcement learning (RL). GBRL is implemented in C++/CUDA aimed to seamlessly integrate within popular RL libraries. 

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

.. note::

   Monotonic constraints apply to the output dimensions covered by the optimizer
   (``start_idx`` to ``stop_idx-1``). They require **SGD**: GBRL raises a ``ValueError``
   if *any* optimizer on the model uses Adam, and ``MultiGBTLearner`` does not support
   them at all. See :doc:`examples` for details.

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   quickstart
   examples

.. toctree::
   :maxdepth: 2
   :caption: Documentation:
   :hidden:

   models/index
   learners/index

