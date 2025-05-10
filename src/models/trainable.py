from .gcn import *

class Trainable:
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(data.x, data.edge_index)
        loss = self.criterion(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test(self, data, type='test'):
        self.model.eval()
        
        with torch.no_grad():
            node_mask = []
            output = self.model(data.x, data.edge_index)
            if type=='test':
                node_mask = data.test_mask
            elif type=='val':
                node_mask = data.val_mask
            
            val_loss = self.criterion(output[node_mask], data.y[node_mask]).item()
            pred = output.argmax(dim=1)
            correct = (pred[node_mask] == data.y[node_mask]).sum().item()
            accuracy = correct / node_mask.sum().item()
        return val_loss, accuracy

    def fit(self, data, epochs=200):
        for epoch in range(0, epochs + 1):
            train_loss = self.train(data)
            val_loss, val_accuracy = self.test(data, type='val')
            if epoch % 20 == 0:
                print(f'Epoch {epoch}, Train Loss - {train_loss}, Val Loss - {val_loss}, Val Accuracy - {val_accuracy}')