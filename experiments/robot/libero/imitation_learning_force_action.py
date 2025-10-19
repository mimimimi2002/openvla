import os
import numpy as np
import tensorflow as tf
import pickle as pkl
from baselines import logger
from baselines.common import set_global_seeds
from mpi4py import MPI
from datasets import load_dataset
from tensorflow.keras import layers
from tensorflow import keras

# -------------------------------
# Residual Policy Network (Δaction)
# -------------------------------
class ResidualPolicy(tf.keras.Model):
    def __init__(self, input_dim, action_dim, lr=1e-3):
        super(ResidualPolicy, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim

        # ネットワーク定義
        self.model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(action_dim)  # Δactionの出力
        ])

        # オプティマイザと損失
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    @tf.function
    def train_step(self, obs, delta_action):
        with tf.GradientTape() as tape:
            pred = self.model(obs, training=True)
            loss = self.loss_fn(delta_action, pred)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def predict(self, obs):
        obs = np.array(obs, dtype=np.float32)
        return self.model(obs, training=False).numpy()

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)
# -------------------------------
# Data Preparation
# -------------------------------
def prepare_dataset(data):
    obs_concat = np.concatenate([
        data['observation.state'],
        data['observation.states.ee_state'],
        data['observation.states.joint_state'],
        data['observation.states.gripper_state'],
        data['observation.states.finger1_collision'],
        data['observation.states.finger1_pad_collision'],
        data['observation.states.finger2_collision'],
        data['observation.states.finger2_pad_collision'],
    ], axis=1)
    
    actions = np.array(data['action'])

    # delta_action = actions - actions_base
    return obs_concat, actions

# -------------------------------
# Training Loop
# -------------------------------
def train_residual_policy(obs_concat, delta_action, input_dim, action_dim,
                          n_epochs=50, batch_size=64, logdir=None):
    if logdir:
        os.makedirs(logdir, exist_ok=True)
        logger.configure(dir=logdir)

    policy = ResidualPolicy(input_dim=input_dim, action_dim=action_dim, lr=1e-3)

    N = obs_concat.shape[0]
    for epoch in range(n_epochs):
        perm = np.random.permutation(N)
        losses = []
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            loss = policy.train_step(obs_concat[idx], delta_action[idx])
            losses.append(loss)
        mean_loss = np.mean(losses)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {mean_loss:.6f}")
        if logdir:
            logger.record_tabular('epoch', epoch)
            logger.record_tabular('loss', mean_loss)
            logger.dump_tabular()
    return policy

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
  # データセットをロード
  ds = load_dataset("mimimimi2002/openvla_libero_spatial_force")

  # train split を取得
  train_ds = ds["train"]

  valid_force_data = train_ds.filter(lambda x: x["observation.states.finger1_collision"] != [0, 0, 0, 0, 0, 0] and x["observation.states.finger1_pad_collision"] != [0, 0, 0, 0, 0, 0] and x["observation.states.finger2_collision"] != [0, 0, 0, 0, 0, 0] and x["observation.states.finger2_pad_collision"] != [0, 0, 0, 0, 0, 0])

  obs_concat, delta_action = prepare_dataset(valid_force_data)

  input_dim = obs_concat.shape[1]
  action_dim = delta_action.shape[1]

  policy = train_residual_policy(obs_concat, delta_action,
                                  input_dim=input_dim, action_dim=action_dim,
                                  n_epochs=100, batch_size=128,
                                  logdir="./residual_policy_logs")

  # 保存
  policy.save("./residual_policy.pkl")
