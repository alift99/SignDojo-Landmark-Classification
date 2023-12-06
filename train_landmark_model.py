import torch
from torch import nn
from dataloader import LandmarkFeatures
from torch.utils.data import DataLoader
from tqdm import tqdm


class LinearClassifier(nn.Module):
    def __init__(self, input_dim=42, hidden_dim=128, output_dim=26):
        super(LinearClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearClassifier(input_dim=42, hidden_dim=128, output_dim=26).to(device)

    train_dataset = LandmarkFeatures(filename='datasets/train.csv')
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    valid_dataset = LandmarkFeatures(filename='datasets/valid.csv')
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_dataset = LandmarkFeatures(filename='datasets/test.csv')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_avg_eval_loss = 1000000
    best_model = None
    for epoch in range(30):
        train_loss = 0
        model.train()
        for inputs, targets in tqdm(train_loader):
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)

            preds = model(inputs)
            loss = loss_fn(preds, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss

        eval_loss = 0
        model.eval()
        for inputs, targets in tqdm(valid_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                preds = model(inputs)

                loss = loss_fn(preds, targets)
            eval_loss += loss


        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_eval_loss = eval_loss / len(valid_loader.dataset)
        print(f'Train loss at epoch {epoch}: {avg_train_loss}')
        print(f'Eval loss at epoch: {epoch}: {avg_eval_loss}')
        if avg_eval_loss < best_avg_eval_loss:
            best_avg_eval_loss = avg_eval_loss
            best_model = model
            print('New best model found. Saving model...')

    # Eval loop
    best_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = best_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    torch.save(model.state_dict(), 'saved_models/linear_model.pth')


if __name__ == '__main__':
    train()


