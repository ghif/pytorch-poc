import os
import json

import torch
import torch.nn.functional as F

from tqdm import tqdm
from timeit import default_timer as timer
from time import process_time

def train(dataloader, model, loss_fn, optimizer, device=None):
    if device is None:
        device = "cpu"

    model = model.to(device)
    model.train() # set model to training mode

    train_time = 0.
    with tqdm(dataloader, unit="batch") as tepoch:
        for (X, y) in tepoch:
            start_t = process_time()
            X, y = X.to(device), y.to(device)
            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            elapsed_time = process_time() - start_t
            train_time += elapsed_time

            tepoch.set_postfix({
                'loss': loss.item(),
                'time(secs)': train_time
            })
        # end for
    
    return train_time

def test(dataloader, model, loss_fn, device=None):
    if device is None:
        device = "cpu"

    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model = model.to(device)
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
    print(f"- Accuracy: {(100*accuracy):.2f}%, Avg loss: {loss:.4f} \n")

    return loss, accuracy

def evaluate(model, dataloader, device="cpu"):
    """
    Evaluate the performance of a model on a given dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The dataloader for the evaluation dataset.
        device (str, optional): The device to use for evaluation. Defaults to "cpu".

    Returns:
        tuple: A tuple containing the average loss, accuracy, and evaluation time.

    """
    print(f"Evaluation -- using device {device}")

    model.eval()
    eval_time = 0.0
    losses = 0.0
    correct = 0
    
    with tqdm(dataloader, unit="batch") as tepoch:
        for (X, y) in tepoch:
            start_t = process_time()
            X, y = X.to(device), y.to(device)
            
            # Compute loss
            with torch.no_grad():
                logits = model(X)
            loss = F.cross_entropy(logits, y)
            batch_size = X.shape[0]
            losses += (loss.item() / batch_size)

            # Compute correct predictions
            pred = torch.argmax(logits, axis=1)
            correct += torch.sum(pred == y)

            elapsed_time = process_time() - start_t
            eval_time += elapsed_time

            tepoch.set_postfix({
                'loss': loss.item(),
                'time(secs)': eval_time
            })
        # end for
            
    model.train()

    num_data = len(dataloader.dataset)
    accuracy = correct.item() / num_data
    return losses, accuracy, eval_time


def fit(model, 
        train_dataloader, 
        test_dataloader, 
        loss_fn, 
        optimizer, 
        n_epochs=10, 
        checkpoint_dir=None, 
        model_name=None,
        writer=None, 
        device=None
    ):
    history = {
        "train_losses": [],
        "test_losses": [],
        "train_accs": [],
        "test_accs": [],
        "train_times": []
    }
    
    for t in range(n_epochs):
        print(f"Epoch {t+1}\n-------------------------------")

        # Train!
        train_time = train(train_dataloader, model, loss_fn, optimizer, device=device)

        # Evaluate training and testing performance
        print(f"Training performance:")
        train_loss, train_acc = test(train_dataloader, model, loss_fn, device=device) # Training performance
        print(f"Test performance:")
        test_loss, test_acc = test(test_dataloader, model, loss_fn, device=device) # Testing performance

        print(f"Elapsed time: {train_time:.2f} seconds\n")

        history["train_losses"].append(train_loss)
        history["test_losses"].append(test_loss)
        history["train_accs"].append(train_acc)
        history["test_accs"].append(test_acc)
        history["train_times"].append(train_time)

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
        # end if writer
            
        if checkpoint_dir is not None:
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            
            # Save model
            model_path = os.path.join(checkpoint_dir, f"{model_name}.pth")
            torch.save(model.state_dict(), model_path)
            
            # Save training history
            history_path = os.path.join(checkpoint_dir, f"{model_name}_hist.json")
            
            json_object = json.dumps(history) # serializing json
            with open(history_path, "w") as outfile:
                outfile.write(json_object) # write to json file
                
            print(f"Saved PyTorch Model State to {model_path} and history xto {history_path}")
            
        # end if checkpoint
            
    # end for

    if writer is not None:        
        writer.flush()
        writer.close() 

    return history