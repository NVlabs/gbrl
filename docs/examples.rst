Examples
========

.. important::

    These examples are only to demonstrate the use of the GBRL library and its functions. Specific algorithm implementations can be found for stable_baselines3 in `the GBRL_SB3 repository <https://github.com/NVlabs/gbrl_sb3>`_.

The full tutorial is also available as a `jupyter notebook <https://github.com/NVlabs/gbrl/blob/master/tutorial.ipynb>`_

Basic Usage: Training, Saving, Loading
--------------------------------------
In the following example, we will train, save, and load a GBRL model incrementally.
We will use the base `GradientBoostingTrees` class and get familiarized with the basic usage of the GBRL library.
We will train a GBRL model as in supervised learning on the `Diabetes dataset from sklearn <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html#sklearn.datasets.load_diabetes>`_.

Basic imports and preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    import torch as th
    import gymnasium as gym 

    from sklearn import datasets
    from torch.nn.functional import mse_loss 
    from torch.distributions import Categorical

    from gbrl import GradientBoostingTrees, cuda_available, ParametricActor

Pre-process data
~~~~~~~~~~~~~~~~

.. code-block:: python

    # incremental learning dataset
    X_numpy, y_numpy = datasets.load_diabetes(return_X_y=True, as_frame=False, scaled=False)
    out_dim = 1 if len(y_numpy.shape) == 1 else y_numpy.shape[1]
    if out_dim == 1:
        y_numpy = y_numpy[:, np.newaxis]

    X, y = th.tensor(X_numpy, dtype=th.float32), th.tensor(y_numpy, dtype=th.float32)

Setting up a GBRL model
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # CUDA is not deterministic
    device = 'cuda' if cuda_available else 'cpu'

    # initializing model parameters
    tree_struct = {'max_depth': 4, 
                   'n_bins': 256,
                   'min_data_in_leaf': 0,
                   'par_th': 2,
                   'grow_policy': 'oblivious'}

    optimizer = {'algo': 'SGD',
                 'lr': 1.0}

    gbrl_params = {
                   "split_score_func": "Cosine",
                   "generator_type": "Quantile"
                  }

    # setting up model
    gbt_model = GradientBoostingTrees(
                        output_dim=out_dim,
                        tree_struct=tree_struct,
                        optimizer=optimizer,
                        gbrl_params=gbrl_params,
                        verbose=0,
                        device=device)
    gbt_model.set_bias_from_targets(y)

Incremental learning
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # training for 10 epochs
    n_epochs = 10
    for _ in range(n_epochs):
        # forward pass - setting requires_grad=True is mandatory for training
        # y_pred is a torch tensor
        y_pred = gbt_model(X, requires_grad=True)
        # calculate loss - we must scale pytorch's mse loss function by 0.5 to get the correct MSE gradient
        loss = 0.5 * mse_loss(y_pred, y)
        loss.backward()
        # perform a boosting step
        gbt_model.step(X)
        print(f"Boosting iteration: {gbt_model.get_iteration()} RMSE loss: {loss.sqrt()}")

GBT work with per-sample gradients but pytorch typically calculates the expected loss. GBRL internally multiplies the gradients with the number of samples when calling the step function. Therefore, when working with pytorch losses and multi-output targets one should take this into consideration.  
For example: when using a summation reduction

.. code-block:: python

    gbt_model = GradientBoostingTrees(
                        output_dim=out_dim,
                        tree_struct=tree_struct,
                        optimizer=optimizer,
                        gbrl_params=gbrl_params,
                        verbose=0,
                        device=device)
    gbt_model.set_bias_from_targets(y)
    # continuing training 10 epochs using a sum reduction
    n_epochs = 10
    for _ in range(n_epochs):
        y_pred = gbt_model(X, requires_grad=True)
        # we divide the loss by the number of samples to compensate for GBRL's built-in multiplication by the same value   
        loss = 0.5 * mse_loss(y_pred, y, reduction='sum') / len(y_pred) 
        loss.backward()
        # perform a boosting step
        gbt_model.step(X)
        print(f"Boosting iteration: {gbt_model.get_iteration()} RMSE loss: {loss.sqrt()}")

or when working with multi-dimensional outputs

.. code-block:: python

    y_multi = th.concat([y, y], dim=1)
    out_dim = y_multi.shape[1]
    gbt_model = GradientBoostingTrees(
                        output_dim=out_dim,
                        tree_struct=tree_struct,
                        optimizer=optimizer,
                        gbrl_params=gbrl_params,
                        verbose=0,
                        device=device)
    gbt_model.set_bias_from_targets(y_multi)
    # continuing training 10 epochs using a sum reduction
    n_epochs = 10
    for _ in range(n_epochs):
        y_pred = gbt_model(X, requires_grad=True)
        # we multiply the loss by the output dimension to compensate for pytorch's mean reduction for MSE loss that averages across all dimensions.
        # this step is necessary to get the correct loss gradient - however the loss value itself is correct
        loss = 0.5 * mse_loss(y_pred, y_multi) * out_dim
        loss.backward()
        # perform a boosting step
        gbt_model.step(X)
        print(f"Boosting iteration: {gbt_model.get_iteration()} RMSE loss: {(loss / out_dim).sqrt()}")

Saving and Loading a GBRL Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Saving and loading in GBRL is straightforward.

.. code-block:: python

    # Call the save_model method of a GBRL class
    # GBRL will automatically save the file with the .gbrl_model ending
    # The file will be saved in the current working directory
    # Provide the absolute path to save the file in a different directory.
    gbt_model.save_model('gbt_model_tutorial')
    # Loading a saved model is similar and is done by calling the specific class instance.
    loaded_gbt_model = GradientBoostingTrees.load_model('gbt_model_tutorial')

    y_save = gbt_model(X)
    y_load = loaded_gbt_model(X)

Manually Calculated Gradients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Alternatively, GBRL can use manually calculated gradients. Calling the `predict` method instead of the `__call__` method, returns a numpy array instead of a PyTorch tensor. Autograd libraries or manual calculations can be used to calculate gradients.  
Fitting manually calculated gradients is done using the `_model.step` method that receives numpy arrays. 

.. code-block:: python
    
    # initializing model parameters
    tree_struct = {'max_depth': 4, 
                'n_bins': 256,
                'min_data_in_leaf': 0,
                'par_th': 2,
                'grow_policy': 'oblivious'}
                
    optimizer = { 'algo': 'SGD',
                'lr': 1.0}

    gbrl_params = {
                "split_score_func": "Cosine",
                "generator_type": "Quantile"}

    # setting up model
    gbt_model = GradientBoostingTrees(
                        output_dim=1,
                        tree_struct=tree_struct,
                        optimizer=optimizer,
                        gbrl_params=gbrl_params,
                        verbose=0,
                        device=device)
    # works with numpy arrays as well as PyTorch tensors
    gbt_model.set_bias_from_targets(y_numpy)
    # training for 10 epochs
    n_epochs = 10
    for _ in range(n_epochs):
        # y_pred is a numpy array
        y_pred = gbt_model.predict(X_numpy)
        loss = np.sqrt(0.5 * ((y_pred - y_numpy)**2).mean())
        grads = y_pred - y_numpy
        # perform a boosting step
        gbt_model._model.step(X_numpy, grads)
        print(f"Boosting iteration: {gbt_model.get_iteration()} RMSE loss: {loss}")

Multiple boosting iterations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
GBRL supports training multiple boosting iterations with targets similar to other GBT libraries. This is done using the `fit` method.  

.. important::

    Only the RMSE loss function is supported for the `fit` method

.. code-block:: python

    gbt_model = GradientBoostingTrees(
                        output_dim=1,
                        tree_struct=tree_struct,
                        optimizer=optimizer,
                        gbrl_params=gbrl_params,
                        verbose=1,
                        device=device)
    final_loss = gbt_model.fit(X_numpy, y_numpy, iterations=10)

RL using GBRL
-------------
Now that we have seen how GBRL is trained using incremental learning and PyTorch, we can use it within an RL training loop.

Let's start by training a simple Reinforce algorithm.

.. code-block:: python
    
    def calculate_returns(rewards, gamma):
        returns = []
        running_g = 0.0
        for reward in rewards[::-1]:
            running_g = reward + gamma * running_g
            returns.insert(0, running_g)
        return returns

    env = gym.make("CartPole-v1")
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward
    num_episodes = 1000
    gamma = 0.99
    optimizer = { 'algo': 'SGD',
                'lr': 0.05}

    bias = np.zeros(env.action_space.n, dtype=np.single)
    agent = ParametricActor(
                        output_dim=env.action_space.n,
                        tree_struct=tree_struct,
                        policy_optimizer=optimizer,
                        gbrl_params=gbrl_params,
                        verbose=0,
                        bias=bias, 
                        device='cpu')

    update_every = 20

    rollout_buffer = {'actions': [], 'obs': [], 'returns': []}
    for episode in range(num_episodes):
        # gymnasium v26 requires users to set seed while resetting the environment
        obs, info = wrapped_env.reset(seed=0)
        rollout_buffer['rewards'] = []

        done = False
        while not done:
            action_logits = agent(obs)
            action = Categorical(logits=action_logits).sample()
            action_numpy = action.cpu().numpy()
            
            obs, reward, terminated, truncated, info = wrapped_env.step(action_numpy.squeeze())
            rollout_buffer['rewards'].append(reward)
            rollout_buffer['obs'].append(obs)
            rollout_buffer['actions'].append(action)

            done = terminated or truncated
        
        rollout_buffer['returns'].extend(calculate_returns(rollout_buffer['rewards'], gamma))

        if episode % update_every == 0 and episode > 0:
            returns = th.tensor(rollout_buffer['returns'])
            actions = th.cat(rollout_buffer['actions'])
            # input to model can be either a torch tensor or a numpy ndarray
            observations = np.stack(rollout_buffer['obs'])
            # model update
            action_logits = agent(observations, requires_grad=True)
            dist = Categorical(logits=action_logits)
            log_probs = dist.log_prob(actions)
            # calculate reinforce loss with subtracted baseline
            loss = -(log_probs * (returns - returns.mean())).mean()
            loss.backward()
            grads = agent.step(observations)
            rollout_buffer = {'actions': [], 'obs': [], 'returns': []}

        if episode % 100 == 0:
            print(f"Episode {episode} - boosting iteration: {agent.get_iteration()} episodic return: {np.mean(wrapped_env.return_queue)}")
