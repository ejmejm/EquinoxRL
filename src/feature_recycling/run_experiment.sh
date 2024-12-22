python feature_recycling.py --config-name=nonlinear_geoff.yaml \
device=cpu \
seed=0 \
\
task.n_real_features=10 \
task.n_features=10 \
task.weight_scale=0.5 \
task.flip_rate=0 \
task.n_layers=2 \
task.hidden_dim=20 \
task.sparsity=0.8 \
\
feature_recycling.distractor_chance=0.0 \
feature_recycling.recycle_rate=0.0 \
feature_recycling.use_cbp_utility=true \
feature_recycling.feature_protection_steps=100 \
feature_recycling.utility_decay=0.999 \
\
train.optimizer=idbd \
train.batch_size=1 \
train.total_steps=50000 \
train.log_freq=100 \
train.learning_rate=0.005 \
\
model.n_layers=2 \
model.hidden_dim=128 \
model.weight_init_method=zeros \
\
idbd.meta_learning_rate=0.1 \
idbd.diagonal_approx=false \
\
wandb=false