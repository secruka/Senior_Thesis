import cv2
import numpy as np

bgr = cv2.imread("/Users/ruka/ResNet_proj/clock_kaggle/train/1-00/0.jpg")

def extract_hand_mask(bgr):
    h, w = bgr.shape[:2]
    cx, cy = w//2, h//2

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # 暗い部分を1に（Otsu）
    _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 小ノイズ除去
    bin_inv = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=1)

    # 連結成分
    num, labels = cv2.connectedComponents(bin_inv)

    # 中心seed（半径は針の太さに合わせて調整）
    seed = np.zeros((h,w), np.uint8)
    cv2.circle(seed, (cx,cy), 8, 1, -1)

    seed_labels = labels[(seed==1) & (bin_inv==255)]
    if len(seed_labels) == 0:
        # 針が中心から少しズレてる等の保険
        cv2.circle(seed, (cx,cy), 14, 1, -1)
        seed_labels = labels[(seed==1) & (bin_inv==255)]

    hand_ids = np.unique(seed_labels)
    hand_mask = np.isin(labels, hand_ids).astype(np.uint8) * 255

    # 太さを戻す（必要なら）
    hand_mask = cv2.dilate(hand_mask, np.ones((3,3),np.uint8), iterations=1)
    return hand_mask

def make_views(bgr):
    hand_mask = extract_hand_mask(bgr)

    # 針用
    hand_img = cv2.bitwise_and(bgr, bgr, mask=hand_mask)

    # 文字盤用（針を消す）
    dial_img = cv2.inpaint(bgr, hand_mask, 3, cv2.INPAINT_TELEA)
    return hand_img, dial_img, hand_mask

hand_img, dial_img, hand_mask = make_views(bgr)
cv2.imwrite('/Users/ruka/ResNet_proj/masked/masked_dial.jpg', dial_img)
cv2.imwrite('/Users/ruka/ResNet_proj/masked/masked_hand.jpg', hand_img)
cv2.imwrite('/Users/ruka/ResNet_proj/masked/maske.jpg', hand_mask)