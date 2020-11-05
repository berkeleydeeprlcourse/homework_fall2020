# HW3 Paste-Ins
There's multiple ways to implement the code inside `MLP_policy.py`.

## `policies/MLP_policy.py`
### MLPPolicy
```
def get_action(self, obs: np.ndarray) -> np.ndarray:
    if len(obs.shape) > 1:
        observation = obs
    else:
        observation = obs[None]
    observation = ptu.from_numpy(observation)
    action_distribution = self(observation)
    action = action_distribution.sample()  # don't bother with rsample
    return ptu.to_numpy(action)
def forward(self, observation: torch.FloatTensor):
    if self.discrete:
        logits = self.logits_na(observation)
        action_distribution = distributions.Categorical(logits=logits)
        return action_distribution
    else:
        batch_mean = self.mean_net(observation)
        scale_tril = torch.diag(torch.exp(self.logstd))
        batch_dim = batch_mean.shape[0]
        batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
        action_distribution = distributions.MultivariateNormal(
            batch_mean,
            scale_tril=batch_scale_tril,
        )
        return action_distribution

def forward(self, observation: torch.FloatTensor):
    if self.discrete:
        logits = self.logits_na(observation)
        action_distribution = distributions.Categorical(logits=logits)
        return action_distribution
    else:
        batch_mean = self.mean_net(observation)
        scale_tril = torch.diag(torch.exp(self.logstd))
        batch_dim = batch_mean.shape[0]
        batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
        action_distribution = distributions.MultivariateNormal(
            batch_mean,
            scale_tril=batch_scale_tril,
        )
        return action_distribution
```

## `infrastructure/rl_trainer.py`
```
def collect_training_trajectories(self, itr, initial_expertdata, collect_policy, num_transitions_to_sample, save_expert_data_to_disk=False):
    if itr == 0:
        if initial_expertdata is not None:
            paths = pickle.load(open(self.params['expert_data'], 'rb'))
            return paths, 0, None
        if save_expert_data_to_disk:
            num_transitions_to_sample = self.params['batch_size_initial']

    # collect data to be used for training
    print("\nCollecting data to be used for training...")
    paths, envsteps_this_batch = utils.sample_trajectories(self.env, collect_policy, num_transitions_to_sample, self.params['ep_len'])

    # collect more rollouts with the same policy, to be saved as videos in tensorboard
    train_video_paths = None
    if self.logvideo:
        print('\nCollecting train rollouts to be used for saving videos...')
        train_video_paths = utils.sample_n_trajectories(self.env, collect_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

    if save_expert_data_to_disk and itr == 0:
        with open('expert_data_{}.pkl'.format(self.params['env_name']), 'wb') as file:
            pickle.dump(paths, file)

    return paths, envsteps_this_batch, train_video_paths

def train_agent(self):
    all_logs = []
    for train_step in range(self.params['num_agent_train_steps_per_iter']):
        ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(self.params['train_batch_size'])
        train_log = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
        all_logs.append(train_log)
    return all_logs
```

## `infrastruction/utils.py`
```
def sample_trajectory(env, policy, max_path_length, render=False, render_mode=('rgb_array')):
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        if render:  # feel free to ignore this for now
            if 'rgb_array' in render_mode:
                if hasattr(env.unwrapped, 'sim'):
                    if 'track' in env.unwrapped.model.camera_names:
                        image_obs.append(env.unwrapped.sim.render(camera_name='track', height=500, width=500)[::-1])
                    else:
                        image_obs.append(env.unwrapped.sim.render(height=500, width=500)[::-1])
                else:
                    image_obs.append(env.render(mode=render_mode))
            if 'human' in render_mode:
                env.render(mode=render_mode)
                time.sleep(env.model.opt.timestep)
        obs.append(ob)
        ac = policy.get_action(ob)
        ac = ac[0]
        acs.append(ac)
        ob, rew, done, _ = env.step(ac)
        # add the observation after taking a step to next_obs
        next_obs.append(ob)
        rewards.append(rew)
        steps += 1
        # If the episode ended, the corresponding terminal value is 1
        # otherwise, it is 0
        if done or steps > max_path_length:
            terminals.append(1)
            break
        else:
            terminals.append(0)
    return Path(obs, image_obs, acs, rewards, next_obs, terminals)

def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False, render_mode=('rgb_array')):

    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:

        #collect rollout
        path = sample_trajectory(env, policy, max_path_length, render, render_mode)
        paths.append(path)

        #count steps
        timesteps_this_batch += get_pathlength(path)
        print('At timestep:    ', timesteps_this_batch, '/', min_timesteps_per_batch, end='\r')

    return paths, timesteps_this_batch

def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False, render_mode=('rgb_array')):

    paths = []
    for i in range(ntraj):
        # collect rollout
        path = sample_trajectory(env, policy, max_path_length, render, render_mode)
        paths.append(path)

    return paths
```
