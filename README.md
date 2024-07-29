# ICTSurF: Implicit Continuous-Time Survival Function

This is the repository for the implementation of ICTSurF, a deep learning method for survival analysis.

## Usage

Single risk dataset

```python
from ictsurf import dataset

random_state = 1234
np.random.seed(random_state)
_ = torch.manual_seed(random_state)
features, durations, events = get_metabric_dataset_onehot()

features, features_val, durations, durations_val, events, events_val = train_test_split(
            features, durations, events, test_size=0.15, random_state = random_state, stratify = events)

features_train, features_test, durations_train, durations_test, events_train, events_test = train_test_split(
            features, durations, events, test_size=0.15, random_state = random_state, stratify = events)

# remove the samples with durations greater than the maximum duration in the training set
while durations_train.max()<=durations_test.max():
    test_index_max = durations_test.argmax()
    durations_test = deepcopy(np.delete(durations_test, test_index_max))
    features_test = deepcopy(np.delete(features_test, test_index_max, axis = 0))
    events_test = deepcopy(np.delete(events_test, test_index_max))
  
while durations_train.max()<=durations_val.max():
    test_index_max = durations_val.argmax()
    durations_val = deepcopy(np.delete(durations_val, test_index_max))
    features_val = deepcopy(np.delete(features_val, test_index_max, axis = 0))
    events_val = deepcopy(np.delete(events_val, test_index_max))

# scale time 
mean_time = np.mean(durations_train)
durations_train = durations_train/mean_time
durations_val = durations_val/mean_time
durations_test = durations_test/mean_time

# scale features
scaler =  StandardScaler()
features_train = scaler.fit_transform(features_train)
features_val = scaler.transform(features_val)
features_test = scaler.transform(features_test)

# add 1 for time feature
in_features = features_train.shape[1]+1
num_nodes = [64]
num_nodes_res = [64]
time_dim = 16
batch_norm = True
dropout = 0.0
lr = 0.0002
activation = nn.ReLU
output_risk = 1
batch_size = 256
epochs = 10000
n_discrete_time = 50
patience = 10
device = 'cpu'

# defined network
net = MLPTimeEncode(
in_features, num_nodes, num_nodes_res, time_dim=time_dim, batch_norm= batch_norm,
dropout=dropout, activation=activation, output_risk = output_risk).float()

model = ICTSurF(net).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

model.fit(optimizer, features_train, durations_train, events_train,
            features_val, durations_val, events_val,
    n_discrete_time = n_discrete_time, patience = patience, device = device,
    batch_size=batch_size, epochs=epochs, shuffle=True)
```

Example for multiple event dataset can be found in example_multi.ipynb

## Customization

You can custom your own network architecture

```python
# defined your own network
class CustomNet(nn.Module):
    def __init__(self, in_features, output_risk = 1):
        super().__init__()
        self.output_risk = output_risk
        self.linear = nn.Linear(in_features, 1)
  
    def forward(self, input):
  
        # the input in shape of (batch, time_step, in_features)
        # the last features is the time feature
        time_step = input.shape[1]
        input = input.view(-1, input.shape[-1])
        out = self.linear(input)

        if self.output_risk == 1:

            return out.view(-1, time_step)
        return out.reshape(-1, time_step, 1)
net = CustomNet(in_features)

model = ICTSurF(net).to(device)
```

Custom your own discretization scheme

```python


class CTCutEqualSpacing:
    def __init__(self, n_cuts):

        self.n_cuts = n_cuts
        self.horizon = None

    def fit(self, durations, events):
        return self

    def transform(self, durations, events):
        duration_indices = []
        horizons = []
        for duration, event in zip(durations, events):
            idx = self.n_cuts-1

            duration_indices.append(idx)

            horizons.append(np.linspace(0, duration, self.n_cuts))

        duration_indices = np.array(duration_indices)
        horizons = np.array(horizons)
	# duration_indices, events must be 1d
	# horizons must be 2d array of shape (n_samples, horizon_times)
        return duration_indices.astype(np.int64), events.astype(np.int64), horizons

    def fit_transform(self, durations, events):
        self.fit(durations, events)
        return self.transform(durations, events)

model = the_model(net, processor_class = CTCutEqualSpacing)
```

Get survival probability for your own evaluation metric

```python
# defined model and fit the data
model = ICTSurF(net).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

model.fit(optimizer, features_train, durations_train, events_train,
            features_val, durations_val, events_val,
    n_discrete_time = n_discrete_time, patience = patience, device = device,
    batch_size=batch_size, epochs=epochs, shuffle=True)

# select specific time of interest
eval_time = np.quantile(durations_test[events_test == 1], 0.25)

time_of_interests = np.array([eval_time]*len(features_test))
fake_events = np.array([1]*len(features_test))

# create dataloader for evaluation using data processor that already fitted from model
test_loader = get_loader(features_test, time_of_interests, fake_events, model.processor ,batch_size=256, fit_y=False)

preds = test_step(model, test_loader, device = device)

# get hazard
hazard = model.pred_to_hazard(preds)

# get survival probability
# to get survival function we need to integrate the hazard function
# to integrate, we need discretized time from dataloader
# the discretization time can be access from
# test_loader.dataset.extended_data['continuous_times']
surv = model.pred_to_surv(preds, test_loader)

# then using survival probability for your evaluation
```

## Citing and References

```
@article{PUTTANAWARUT2024101531,
title = {ICTSurF: Implicit Continuous-Time Survival Functions with neural networks},
journal = {Informatics in Medicine Unlocked},
year = {2024},
doi = {https://doi.org/10.1016/j.imu.2024.101531},
author = {Chanon Puttanawarut and Panu Looareesuwan and Romen Samuel Wabina and Prut Saowaprut},
}
```
