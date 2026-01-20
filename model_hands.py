import torch
import torch.nn as nn
from torchvision import models, transforms
from deepproblog.train import train_model
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.dataset import DataLoader, Dataset  # これを追加
from deepproblog.engines import ExactEngine
from problog.logic import Term, Constant, Var
from deepproblog.query import Query
from dataset import ClockDataset

class ClockNet(nn.Module):
    def __init__(self, num_classes=12):
        super(ClockNet, self).__init__()
        # 警告が出ていたので、最新の書き方(weights=...)に修正しておきました GPT
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):

        if isinstance(x, list):
            x = torch.stack([item.value for item in x])
            
        # GPU対応: モデルと同じデバイスに転送する
        x = x.to(next(self.parameters()).device)

        return self.resnet(x)

class DeepProbLogDataset(Dataset):
    def __init__(self, pytorch_dataset):
        self.dataset = pytorch_dataset
        self._query_cache = {}  # Lazy-load queries to avoid memory overflow

    def to_query(self, i):
        if i not in self._query_cache:
            img, h, m = self.dataset[i]
            
            # クエリ: time(X, h, m).
            q_term = Term('time', Var('X'), Constant(h), Constant(m))
            
            # 「変数 X の中身は 画像データ(tensor) です」という辞書
            substitution = {Var('X'): Constant(img)}
            
            #クエリと置換辞書をセットにして返す
            self._query_cache[i] = Query(q_term, substitution)
        return self._query_cache[i]

    def __len__(self):
        return len(self.dataset)

def main():
    print("Setting up the model...")

    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")

    # モデルの定義
    cnn_hour = ClockNet(num_classes=12).to(device)
    cnn_minute = ClockNet(num_classes=12).to(device)

    # DeepProbLogのネットワークラッパーに登録
    net_h = Network(cnn_hour, "net_hour", batching=True)
    net_m = Network(cnn_minute, "net_minute", batching=True)
    
    # 最適化手法の設定 Using Adam optimizer
    net_h.optimizer = torch.optim.Adam(cnn_hour.parameters(), lr=1e-4)
    net_m.optimizer = torch.optim.Adam(cnn_minute.parameters(), lr=1e-4)

    # DeepProbLogモデルのロード
    model = Model("models/clock.pl", [net_h, net_m])
    model.set_engine(ExactEngine(model))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_path = "clock_kaggle/"  
    csv_name  = "labels_points.csv"

    # Limit dataset size to avoid memory issues
    pt_train = ClockDataset(data_path, subset="train", csv_name=csv_name, transform=transform, max_samples=1000)
    pt_test  = ClockDataset(data_path, subset="test",  csv_name=csv_name, transform=transform, max_samples=500)

    # DeepProbLog用にラップする
    train_dataset = DeepProbLogDataset(pt_train)

    test_dataset  = DeepProbLogDataset(pt_test)

    # DataLoader作成 - reduced batch size from 16 to 4
    train_loader = DataLoader(train_dataset, batch_size=4)
    test_loader  = DataLoader(test_dataset, batch_size=4)

    print(f"Start training with {len(train_dataset)} examples...")
    if len(train_loader) == 0:
        print("Error: Training data not found. Please check the data path.")
        return

 
    print(f"Start training with {len(train_loader)} examples...")
    train_model(
        model,
        train_loader,
        10,
        test_iter=test_loader,
        log_iter=100, 
        profile=0
    )

    torch.save(cnn_hour.state_dict(), "hour_model.pth")
    torch.save(cnn_minute.state_dict(), "minute_model.pth")
    print("Training finished and models saved.")

if __name__ == "__main__":
    main()