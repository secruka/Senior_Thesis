import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
from deepproblog.query import Query
from problog.logic import Term, Constant

class ClockDataset(Dataset):
    def __init__(self, root_dir, subset='train', transform=None):
        """
        Args:
            root_dir (str): データセットのルートディレクトリ (例: "./data")
            subset (str): 'train', 'test', 'valid' のいずれか
            transform (callable, optional): 画像の前処理
        """
        
        self.root_path = Path(root_dir) / subset
        self.transform = transform
        self.images = []
        self.labels = []

        # ディレクトリを走査して画像をロード
        # フォルダ名がラベルになっている (例: "10-10" -> 10時10分)
        if not self.root_path.exists():
            raise FileNotFoundError(f"Directory not found: {self.root_path}")

        for label_dir in self.root_path.iterdir():
            if label_dir.is_dir():
                # ラベル文字列 ("10-10") を解析
                try:
                    label_str = label_dir.name
                    h_str, m_str = label_str.split('-')
                    
                    # 時 (Hour): 1-12
                    hour = int(h_str)
                    
                    # 分 (Minute): 0-59 -> インデックス 1-12 に変換
                    # 05分->1, 10分->2, ..., 55分->11, 00分->12
                    m_val = int(m_str)
                    if m_val == 0:
                        minute_idx = 12
                    else:
                        minute_idx = m_val // 5
                    
                    # 画像ファイルのパスを保存
                    for img_file in label_dir.glob("*.jpg"): # png等の場合は適宜変更
                        self.images.append(img_file)
                        self.labels.append((hour, minute_idx))
                        
                except ValueError:
                    continue # ラベル形式が合わないフォルダはスキップ

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        hour, minute_idx = self.labels[idx]
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, hour, minute_idx

def get_clock_data(root_dir, subset='train'):
    """
    DeepProbLog用のデータリストを作成して返す関数
    戻り値: List[Query]
    クエリ形式: time(tensor(image), hour, minute).
    """
    
    # ResNet等に合わせた標準的な前処理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ClockDataset(root_dir, subset, transform)
    queries = []
    
    print(f"Loading {subset} data from {root_dir}...")
    
    for i in range(len(dataset)):
        img_tensor, h, m = dataset[i]
        
        # DeepProbLogのクエリを作成
        # time(tensor(data), H, M).
        # 画像テンソルをTerm('tensor', ...)でラップするのが重要
        
        # 注意: 1-12の整数をConstantとして扱う
        q = Query(
            Term('time', 
                 Term('tensor', Constant(img_tensor)), 
                 Constant(h), 
                 Constant(m))
        )
        queries.append(q)
        
    print(f"Loaded {len(queries)} examples.")
    return queries

# テスト実行用
if __name__ == "__main__":
    data_path = "/Users/ruka/ResNet_proj/clock_kaggle" 
    
    try:
        train_data = get_clock_data(data_path, subset='train')
        print(f"Example query: {train_data[0]}")
    except Exception as e:
        print(f"Error: {e}")
        print("データセットのパスを確認してください。")