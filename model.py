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
# ---------------------------------------------------------
# 1. ニューラルネットワークの定義 (ResNet18ベース)
# ---------------------------------------------------------
class ClockNet(nn.Module):
    def __init__(self, num_classes=12):
        super(ClockNet, self).__init__()
        # 警告が出ていたので、最新の書き方(weights=...)に修正しておきました
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # === 【重要修正】 DeepProbLog対応 ===
        # DeepProbLogからは「Constant(画像) のリスト」として渡されてきます。
        # これを PyTorch が読める「1つのバッチテンソル」に変換します。
        if isinstance(x, list):
            # 1. リストの中身(Constant)から .value で画像テンソルを取り出す
            # 2. torch.stack で [Batch, Channel, Height, Width] の形に結合する
            x = torch.stack([item.value for item in x])
            
        # GPU対応: モデルと同じデバイスに転送する
        x = x.to(next(self.parameters()).device)
        # ====================================

        return self.resnet(x)


# ---------------------------------------------------------
# 追加: リストをデータセットとして扱うためのラッパー (修正版)
# ---------------------------------------------------------
class DeepProbLogDataset(Dataset):
    def __init__(self, pytorch_dataset):
        self.dataset = pytorch_dataset

    def to_query(self, i):
        img, h, m = self.dataset[i]
        
        # 1. 画像が入る場所を「変数 X」にしておく
        # クエリ: time(X, h, m).
        q_term = Term('time', Var('X'), Constant(h), Constant(m))
        
        # 2. 「変数 X の中身は 画像データ(tensor) です」という辞書を作る
        # substitution = { X : tensor(img) }
        substitution = {Var('X'): Constant(img)}
        
        # 3. クエリと置換辞書をセットにして返す
        return Query(q_term, substitution)

    def __len__(self):
        return len(self.dataset)
# ---------------------------------------------------------
# 2. DeepProbLogモデルの構築関数
# ---------------------------------------------------------
def main():
    print("Setting up the model...")
    # ---------------------------------------------------------
# 追加: リストをデータセットとして扱うためのラッパー
# ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. モデルの定義
    # .to(device) でGPUに転送
    cnn_hour = ClockNet(num_classes=12).to(device)
    cnn_minute = ClockNet(num_classes=12).to(device)
    # ネットワークのインスタンス化
    # 短針用(net_hour)と長針用(net_minute)で2つ用意します
    # (共有バックボーンにする高度な方法もありますが、まずは別々の方が学習が安定します)

    # DeepProbLogのネットワークラッパーに登録
    # .plファイル内の nn(net_hour, ...) と名前を一致させる必要があります
    net_h = Network(cnn_hour, "net_hour", batching=True)
    net_m = Network(cnn_minute, "net_minute", batching=True)
    
    # 最適化手法の設定 (Learning Rateは調整が必要かもしれません)
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
    # ---------------------------------------------------------
    # 3. データの読み込み
    # ---------------------------------------------------------
    # dataset.py のパスに合わせて変更してください
    data_path = "/Users/ruka/ResNet_proj/clock_kaggle"  # 例: "./archive" など
    
    # A. PyTorch標準のDatasetを作る (この時点では画像は読み込まれない)
    pt_train = ClockDataset(data_path, subset='train', transform=transform)
    pt_test  = ClockDataset(data_path, subset='test',  transform=transform)

    # B. DeepProbLog用にラップする
    # ここで渡すのは pt_train (PyTorch Dataset) です。リストではありません。
    train_dataset = DeepProbLogDataset(pt_train)
    test_dataset  = DeepProbLogDataset(pt_test)

    # C. DataLoader作成
    train_loader = DataLoader(train_dataset, batch_size=16)
    test_loader  = DataLoader(test_dataset, batch_size=16)

    print(f"Start training with {len(train_dataset)} examples...")
    # データセットが空でないか確認
    if len(train_loader) == 0:
        print("Error: Training data not found. Please check the data path.")
        return

    # ---------------------------------------------------------
    # 4. 学習の実行
    # ---------------------------------------------------------
    print(f"Start training with {len(train_loader)} examples...")
    # DeepProbLogの学習ループ
    # epochs: エポック数
    # batch_size: バッチサイズ (GPUメモリに合わせて調整)
    train_model(
        model,
        train_loader,
        10,
        test_iter=test_loader, # テストデータで精度確認
        log_iter=100, # 100バッチごとにログを表示
        profile=0
    )

    # モデルの保存 (任意)
    torch.save(cnn_hour.state_dict(), "hour_model.pth")
    torch.save(cnn_minute.state_dict(), "minute_model.pth")
    print("Training finished and models saved.")

if __name__ == "__main__":
    main()