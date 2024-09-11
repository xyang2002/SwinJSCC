import swanlab
import random

# start a new swanlab run to track this script
swanlab.init(
  # set the swanlab project where this run will be logged
  project="my-awesome-project",
  
  # track hyperparameters and run metadata
  config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10
  }
)

# simulate training
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
  acc = 1 - 2 ** -epoch - random.random() / epoch - offset
  loss = 2 ** -epoch + random.random() / epoch + offset

  # log metrics to swanlab
  swanlab.log({"acc": acc, "loss": loss})

# [optional] finish the swanlab run, necessary when using notebooks
swanlab.finish()
