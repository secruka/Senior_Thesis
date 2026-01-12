import torch
import torch.nn as nn

#新規追加、データ学習のため
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.optim as optim


class block(nn.Module):
    def __init__(self, first_conv_in_channels, first_conv_out_channels, identity_conv=None, stride=1):
        """
        残差ブロックを作成するクラス
        Args:
            first_conv_in_channels : 1番目のconv層（1×1）のinput channel数
            first_conv_out_channels : 1番目のconv層（1×1）のoutput channel数
            identity_conv : channel数調整用のconv層
            stride : 3×3conv層におけるstride数。sizeを半分にしたいときは2に設定
        """      
        super(block, self).__init__()

        # 1番目のconv層（1×1）
        #conv2とは何か、3との違いは？
        # ここでの conv1/conv2/conv3 は単に
        # **「ブロック内の1層目・2層目・3層目」**という意味の番号。
        # conv1（1×1）：チャンネル圧縮（計算量削減）
        # conv2（3×3）：空間特徴抽出の本体
        # conv3（1×1）：チャンネル拡大（元の幅に戻す）
        # 番号の意味は「順番」で、機能は上みたいに違う。


        # 1番目の 1×1 conv
        self.conv1 = nn.Conv2d(
            first_conv_in_channels, first_conv_out_channels, kernel_size=1, stride=1, padding=0)
        #バッチ正規化？
        #ResNetは Convの直後にBNが標準セット
        self.bn1 = nn.BatchNorm2d(first_conv_out_channels)

        # 2番目のconv層（3×3）
        # 役割：空間方向の特徴抽出（process）
        # パターン3の時はsizeを変更できるようにstrideは可変
        self.conv2 = nn.Conv2d(
            first_conv_out_channels, first_conv_out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(first_conv_out_channels)

        # 3番目のconv層（1×1）
        # 役割：チャンネル数を4倍に戻して出力する（expand）
        # output channelはinput channelの4倍になる→なぜ？4倍にしたいのか、構造上そうしないと都合が悪いのか
        # conv1 で 1/4 に圧縮して軽くしたぶん、conv3 で 4倍に戻して 表現力を確保してる
        self.conv3 = nn.Conv2d(
            first_conv_out_channels, first_conv_out_channels*4, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(first_conv_out_channels*4)

        #relu
        self.relu = nn.ReLU()

        #これがF(x) + x の+xに該当する部分
        # identityのchannel数の調整が必要な場合はconv層（1×1）を用意、不要な場合はNone
        self.identity_conv = identity_conv


    def forward(self, x):

        identity = x.clone()

        x = self.conv1(x) # 1×1の畳み込み
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x) # 3×3の畳み込み（パターン3(conv3のこと？）の時はstrideが2になるため、ここでsizeが半分になる）
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x) # 1×1の畳み込み
        x = self.bn3(x)

        # 必要な場合はconv層（1×1）を通してidentityのchannel数の調整してから足す
        if self.identity_conv is not None:
            identity = self.identity_conv(identity)
        x += identity

        x = self.relu(x)

        return x
    


############################## ここからresnetの実装 ################################

class ResNet(nn.Module):
    """
    ResNet50の構造は
    conv1（7×7）＋maxpool
    conv2_x を Bottleneck block 3個
    conv3_x を 4個
    conv4_x を 6個
    conv5_x を 3個
    global average pool
    fc（分類）
    という [3,4,6,3] の積み重ね。
    """

    def __init__(self, block, num_classes):
        super(ResNet, self).__init__()

        # conv1はアーキテクチャ通りにベタ打ち    
        # 入力RGB(3ch) → 64ch へ
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        #kernel_sizeを6にしてもいいのか
        # 偶数カーネルは画像の中心を特定できないからpoolingできない
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        # さらに空間サイズを半分（112→56）
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # conv2_xはサイズの変更は不要のため、strideは1
        # blockの後の3は何を表している？→ここのレイヤーで使う残差の数
        self.conv2_x = self._make_layer(block, 3, res_block_in_channels=64, first_conv_out_channels=64, stride=1)

        # conv3_x以降はサイズの変更をする必要があるため、strideは2

        self.conv3_x = self._make_layer(block, 4, res_block_in_channels=256,  first_conv_out_channels=128, stride=2)
        self.conv4_x = self._make_layer(block, 6, res_block_in_channels=512,  first_conv_out_channels=256, stride=2)
        self.conv5_x = self._make_layer(block, 3, res_block_in_channels=1024, first_conv_out_channels=512, stride=2)

        #fcとは何→Fully Connected layer（全結合層）
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4, num_classes)

    def forward(self,x):

        x = self.conv1(x)   # in:(3,224*224)、out:(64,112*112)
        x = self.bn1(x)     # in:(64,112*112)、out:(64,112*112)
        x = self.relu(x)    # in:(64,112*112)、out:(64,112*112)
        x = self.maxpool(x) # in:(64,112*112)、out:(64,56*56)

        x = self.conv2_x(x)  # in:(64,56*56)  、out:(256,56*56)
        x = self.conv3_x(x)  # in:(256,56*56) 、out:(512,28*28)
        x = self.conv4_x(x)  # in:(512,28*28) 、out:(1024,14*14)
        x = self.conv5_x(x)  # in:(1024,14*14)、out:(2048,7*7)
        x = self.avgpool(x)  # # (2048,7,7) -> (2048,1,1)
        x = x.reshape(x.shape[0], -1) # # -> (2048)
        x = self.fc(x) # # -> (num_classes)

        return x
    
    def _make_layer(self, block, num_res_blocks, res_block_in_channels, first_conv_out_channels, stride):
        layers = []

        # 1つ目の残差ブロックではchannel調整、及びsize調整が発生する
        # identifyを足す前に1×1のconv層を追加し、サイズ調整が必要な場合はstrideを2に設定
        identity_conv = nn.Conv2d(res_block_in_channels, first_conv_out_channels*4, kernel_size=1,stride=stride)
        layers.append(block(res_block_in_channels, first_conv_out_channels, identity_conv, stride))

        # 2つ目以降のinput_channel数は1つ目のoutput_channelの4倍
        in_channels = first_conv_out_channels*4

        # channel調整、size調整は発生しないため、identity_convはNone、strideは1
        for i in range(num_res_blocks - 1):
            layers.append(block(in_channels, first_conv_out_channels, identity_conv=None, stride=1))

        return nn.Sequential(*layers)
    




def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # 勾配の初期化
        optimizer.zero_grad() #前のステップで溜まっている勾配をゼロにする

        # 順伝播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 逆伝播 & パラメータ更新
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def main():
    # ===== 0. device の設定 =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #cpuかgpuどちらを使うか
    print("Using device:", device)

    # ===== 1. データの前処理・ロード =====
    # ResNet 用の基本的な前処理（ImageNet と同じ感じ）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 入力サイズを 224x224 に揃える
        transforms.ToTensor(), #テンソル化
        transforms.Normalize( #正規化
            mean=[0.485, 0.456, 0.406],  # ImageNet の平均
            std=[0.229, 0.224, 0.225]    # ImageNet の分散
        )
    ])

    # もしcsv使いたいならgetitemメソッドを使う
    train_dataset = ImageFolder(root="/Users/ruka/ResNet_proj/clock_kaggle/test", transform=transform) # TRANSFORMとは？
    val_dataset   = ImageFolder(root="/Users/ruka/ResNet_proj/clock_kaggle/test",   transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=2) 
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=2)

    # クラス数をデータから自動で取得 → つまり、ここではcsvファイルがなくても分類している
    num_classes = len(train_dataset.classes) #クラスの合計
    print("num_classes:", num_classes) 
    print("classes:", train_dataset.classes)

    # ===== 2. モデルの用意 =====
    model = ResNet(block, num_classes)
    model.to(device) # 処理媒体に従って

    # ===== 3. 損失関数 & Optimizer =====
    criterion = nn.CrossEntropyLoss() #交差エントリー誤差
    optimizer = optim.Adam(model.parameters(), lr=1e-3) # lrとはlearning rate

    # ===== 4. 学習ループ =====
    num_epochs = 5  # とりあえず 10 epoch 動かしてみる

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        #model =　モデル選択、train_loader = データをテンソル化したもの, criteria = 誤差の種類、optimizer = アダム、device = cpu or gpu
        
        # 簡単な検証
        model.eval() #検証メソッド
        correct = 0 
        total = 0
        with torch.no_grad(): #with?, 勾配をどう扱うか？
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1) #max_pooling的な？
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        val_acc = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}  Val Acc: {val_acc:.4f}")

    # ===== 5. モデルを保存 =====
    torch.save(model.state_dict(), "resnet_clock.pth")
    #resnet_clock.pthとして保存
    print("モデルを resnet_clock.pth に保存しました")


if __name__ == "__main__":
    main()
