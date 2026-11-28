Split-RL
========

GBRL provides the core tree-learning implementation used in
`Split-RL: Local Conflict Resolution in Reinforcement Learning <https://arxiv.org/abs/2611.34799>`__,
by Benjamin Fuhrer, Chen Tessler, and Gal Dalal, NeurIPS 2026. The Split-RL
algorithm and experiment code is available at
`NVlabs/split_rl <https://github.com/NVlabs/split_rl>`__.

If you use the Split-RL extensions, please cite:

.. code-block:: bibtex

   @inproceedings{
   fuhrer2026splitrl,
   title={Split-{RL}: Local Conflict Resolution in Reinforcement Learning},
   author={Benjamin Fuhrer and Chen Tessler and Gal Dalal},
   booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
   year={2026},
   url={https://arxiv.org/abs/2611.34799}
   }

Use cases and behavior
----------------------

These extensions support RL experiments with competing objectives, including
multiple reward signals and reward/cost tradeoffs. The RL algorithm computes
objective-specific policy gradients and assigns an objective label to each
sample. GBRL uses those gradients and labels when selecting splits and updating
leaves. Rollout collection, losses, and any constraint-management logic belong
to the calling RL algorithm.

Each node stores the proportion of samples assigned to each objective. Split
scores combine objective-specific gradient scores using these proportions and
objective weights. When the mean objective gradients conflict, a multiplicative
penalty reduces scores for splits whose children retain mixed objective labels.
Policy leaf values combine per-objective mean gradients using the leaf's label
proportions and objective weights.

Configuration and inputs
------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Meaning
   * - ``n_objs``
     - Number of objectives; defaults to 1.
   * - ``lambda_objs``
     - One weight per objective; defaults to all ones. Used in split scores
       and multi-objective policy leaf updates.
   * - ``lambda_penalty``
     - Conflict/label-impurity penalty strength; defaults to 1.0. Set to 0.0
       to disable the penalty while retaining objective-weighted updates.
   * - ``obj_labels``
     - A vector of length ``n_samples`` passed to ``step``, containing integer
       objective indices in ``[0, n_objs - 1]``. Float32 arrays containing those
       integer values are supported. Omitted labels assign samples to objective 0.

For a single ensemble, gradients have shape
``(n_objs, n_samples, output_dim)``. With ``n_objs=1``, ordinary
``(n_samples, output_dim)`` gradients remain supported. Every objective block
contains gradients for every observation; labels determine mixture proportions,
rather than selecting a single gradient block per sample.

.. code-block:: python

   import numpy as np
   from gbrl.models import GBTModel

   observations = np.arange(8, dtype=np.float32).reshape(-1, 1)
   labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)
   gradients = np.ones((2, 8, 1), dtype=np.float32)
   gradients[1] = -1.0

   model = GBTModel(
       tree_struct={
           'max_depth': 2,
           'grow_policy': 'greedy',
           'n_bins': 8,
           'min_data_in_leaf': 1,
           'par_th': 100,
       },
       input_dim=1,
       output_dim=1,
       optimizers={'algo': 'SGD', 'lr': 0.1, 'start_idx': 0, 'stop_idx': 1},
       params={'n_objs': 2, 'lambda_objs': [1.0, 1.0], 'lambda_penalty': 0.9},
       device='cpu',
   )
   model.step(X=observations, grads=gradients, obj_labels=labels)
   predictions = model(observations, requires_grad=False, tensor=False)
   objective_proportions = model.learner.predict_densities(observations)

``predict_densities`` returns, for each observation, the average objective
proportions of the leaves it reaches across the trees, so each row sums to 1.
An untrained model returns zeros. These proportions describe which objectives
the training samples came from; they are not calibrated probabilities of
success or safety.

Actor and critics
-----------------

``ActorCritic`` provides a policy and a scalar reward critic.
``CostActorCritic`` adds a scalar cost critic and returns
``(policy, reward_value, cost_value)``.

With a shared tree (``shared_tree_struct=True``), ``output_dim`` is the policy
width plus two, and the three optimizers cover the policy columns, the reward
column, and the cost column. With separate trees, each model's optimizer
indices are relative to that model.

Only the policy uses multiple objectives. Pass critic gradients as usual: the
``step`` methods place them in objective block 0 for you. With separate trees,
``actor_step`` accepts ``obj_labels``, while ``critic_step`` and
``cost_critic_step`` update their critics independently.

``step`` accepts NumPy arrays, PyTorch tensors, or a tuple with one
policy-gradient array per objective. ``CostActorCritic.get_grads()`` returns the
policy, reward, and cost gradients.

Label validation
----------------

GBRL checks the shape of ``obj_labels`` on every call. On CPU it also checks
that every label is a valid objective index. On CUDA, label values are not
checked by default, because checking them would force a GPU-to-CPU
synchronization on every step. To check them anyway, pass
``params={'validate_obj_labels': True, ...}``. Set it to ``False`` to skip the
checks on CPU as well. Either way, labels must be integers in
``[0, n_objs - 1]``; out-of-range labels on unchecked paths are ignored when
counting objective proportions.

Loading older models
--------------------

Models saved with GBRL 1.1.2 through 1.1.9 load as single-objective models
(``n_objs=1``, objective weight 1, no conflict penalty). Saving them again
writes the current format. Models from earlier versions cannot be loaded.
Older model files can only be loaded on the same kind of platform they were
saved on.

Limitations
-----------

- Multi-objective models (``n_objs > 1``) train with ``step`` only; ``fit`` is
  not supported.
- ``control_variates`` is not supported with ``n_objs > 1``.
- Monotonic constraints apply to objective 0 only.
- Multi-objective training is supported on CPU and CUDA, for greedy and
  oblivious trees, with both Cosine and L2 split scores.
