import argparse, os
import torch
from PIL import Image
from torchvision import transforms
import numpy as np

from problog.logic import Term, Var, Constant
from deepproblog.query import Query

from train_clock_integrated_heatmap import build_deepproblog_model
from eval_report_time import naive_decode_time, parse_out_key_to_hour_minute

def topk(probs, k):
    probs = probs.detach().cpu().numpy().astype(float)
    idx = np.argsort(-probs)[:k]
    return [(int(i), float(probs[i])) for i in idx]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prolog", required=True)
    ap.add_argument("--dial_pth", required=True)
    ap.add_argument("--hour_pth", required=True)
    ap.add_argument("--minute_pth", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    img = Image.open(args.image).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    # build DPB model (also gives you the torch nets)
    model, cnn_dial, cnn_hour, cnn_minute = build_deepproblog_model(
        prolog_file=args.prolog,
        device=device,
        lr_dial=1e-4, lr_hands=1e-4,
        weights_dial=args.dial_pth,
        weights_hour=args.hour_pth,
        weights_minute=args.minute_pth,
    )
    model.eval()
    cnn_dial.eval(); cnn_hour.eval(); cnn_minute.eval()

    with torch.no_grad():
        dial_p = cnn_dial(x)[0]
        hour_p = cnn_hour(x)[0]
        min_p  = cnn_minute(x)[0]

    d_pred = int(torch.argmax(dial_p).item())
    h_pred = int(torch.argmax(hour_p).item())
    m_pred = int(torch.argmax(min_p).item())
    naive_h, naive_m = naive_decode_time(d_pred, h_pred, m_pred)

    print("== Module topk ==")
    print("dial(topk):", topk(dial_p, min(args.topk, 4)))
    print("hour_img(topk):", topk(hour_p, args.topk))
    print("minute_img(topk):", topk(min_p, args.topk))
    print("\n== Naive decoded time ==")
    print(f"naive: {naive_h:02d}:{naive_m:02d}")

    # DPB time posterior
    q_time = Term("time", Var("X"), Var("H"), Var("M"))
    img_const = Constant(x[0].detach().cpu())  # DPB needs CPU tensor in Constant
    query = Query(q_time, {Var("X"): img_const}, output_ind=(1,2))
    ans = model.solve([query])[0]

    parsed = []
    for k, p in ans.result.items():
        try:
            hh, mm = parse_out_key_to_hour_minute(k)
            parsed.append((hh, mm, float(p)))
        except Exception:
            pass
    parsed.sort(key=lambda t: t[2], reverse=True)

    print("\n== DPB topk times ==")
    for hh, mm, p in parsed[:args.topk]:
        print(f"{hh:02d}:{mm:02d}  p={p:.4f}")

if __name__ == "__main__":
    main()
