from tqdm import tqdm
import torch
from timeit import default_timer as timer

device = (
    "cuda"
    if torch.cuda.is_available()
    # else "mps"
    # if torch.backends.mps.is_available()
    else "cpu"
)
def train(dataloader, model, loss_fn, optimizer):
    model.train() # set model to training mode
    with tqdm(dataloader, unit="batch") as tepoch:
        for (X, y) in tepoch:
            X, y = X.to(device), y.to(device)
            
            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tepoch.set_postfix(loss=loss.item())
            # if batch % 100 == 0:
            #     loss, current = loss.item(), batch * len(X)
            #     # print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() # set model to evaluation mode
    loss, accuracy = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss += loss_fn(pred, y).item()
            accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()
    loss /= num_batches
    accuracy /= size
    print(f"Test Performance: \n Accuracy: {(100*accuracy):.2f}%, Avg loss: {loss:.4f} \n")

    return loss, accuracy


def fit(model, train_dataloader, test_dataloader, loss_fn, optimizer, n_epochs=10, checkpoint_path=None, writer=None):
    for t in range(n_epochs):
        print(f"Epoch {t+1}\n-------------------------------")

        # Train!
        start_time = timer()
        train(train_dataloader, model, loss_fn, optimizer)
        end_time = timer()

        if checkpoint_path is not None:
            # Save model
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved PyTorch Model State to {checkpoint_path}")

        # Evaluate training and testing performance
        train_loss, train_acc = test(train_dataloader, model, loss_fn) # Training performance
        test_loss, test_acc = test(test_dataloader, model, loss_fn) # Testing performance

        print(f"Elapsed time: {end_time - start_time:.2f} seconds\n")

        if writer is not None:
            # Write loss and accuracy to tensorboard
            writer.add_scalars(
                "Loss",
                {
                    "train": train_loss,
                    "test": test_loss,
                },
                t,
            )
            writer.add_scalars(
                "Accuracy",
                {
                    "train": train_acc,
                    "test": test_acc,
                },
                t,
            )
    # end for

    if writer is not None:        
        writer.flush()
        writer.close() 