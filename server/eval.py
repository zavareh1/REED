import torch

def evaluate_global(adapter, model, dataloader, loss_fn):
    # model is already synchronized via adapter.from_vector before calling this
    model.eval(); n, loss_sum, correct = 0, 0.0, 0
    with torch.no_grad():
        for xb, yb in dataloader:
            logits = model(xb)
            loss = loss_fn(logits, yb).item()
            loss_sum += loss * yb.size(0); n += yb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
    return dict(loss=loss_sum/n, acc=correct/n)
