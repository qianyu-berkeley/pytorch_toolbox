# pytorch_toolbox

The purpose of the repo is the develop a deep learning tool library using pytorch framework and test with some public datasets


## Pytorch Training Loop

```python
def fit():
    for epoch in range(epochs):
        for xb,yb in train_dl:         # iterate data from a dataloader (mini-batchs)
            pred = model(xb)           # calculate predictions
            loss = loss_func(pred, yb) # calculate loss
            loss.backward()            # Calculate gradients
            opt.step()                 # Update with the learning rate
            opt.zero_grad()            # Reset gradient (This is different from tensorflow)
```