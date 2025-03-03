from .gcn import *
import copy

class Trainable:
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.best_model_state = None  # Store best model parameters
        self.best_val_acc = 0.0

    def train(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(data.x, data.edge_index)
        loss = self.criterion(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test(self, data):
        self.model.eval()
        with torch.no_grad():
            output = self.model(data.x, data.edge_index)
            val_loss = self.criterion(output[data.test_mask], data.y[data.test_mask]).item()
            pred = output.argmax(dim=1)
            correct = (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
            accuracy = correct / data.test_mask.sum().item()
        return val_loss, accuracy

    def fit(self, data, epochs=200, select_best=True):
        for epoch in range(0, epochs + 1):
            train_loss = self.train(data)
            val_loss, val_accuracy = self.test(data)

            if val_accuracy > self.best_val_acc:
                self.best_val_acc = val_accuracy
                self.best_model_state = copy.deepcopy(self.model.state_dict())
            
            if epoch % 20 == 0:
                print(f'Epoch {epoch}, Train Loss - {train_loss}, Val Loss - {val_loss}, Val Accuracy - {val_accuracy}')

        if self.best_model_state and select_best:
            self.model.load_state_dict(self.best_model_state)
            print(f"Loaded best model with Val Accuracy: {self.best_val_acc:.4f}")