from pathlib import Path
from PIL import Image, ImageDraw
from collections import Counter
from train import initialize_model, MODEL_NAME
from torchvision import transforms
import torch

START = (1170, 576)
COLUMN_OFFSET = 152
ROW_OFFSET = 31
WIDTH = 20
SUITE_MAP = {
    "box": ("B", 0),
    "chin": ("G", 0),
    "y": ("R", 0),
    "special": ("S", 1),
}


def extract_imgs(file):
    cropped_imgs = []
    with Image.open(file) as img:
        draw = ImageDraw.Draw(img)
        for i in range(5):
            for j in range(8):
                box = (
                    START[0] + j * (COLUMN_OFFSET),
                    START[1] + ROW_OFFSET * i,
                    START[0] + j * (COLUMN_OFFSET) + WIDTH,
                    START[1] + ROW_OFFSET * (i + 1) - 7,
                )
                cropped_imgs.append((img.crop(box), i, j))
                draw.line(box, fill=128)
        # img.show()
    return cropped_imgs


def predict_color(img):
    img = img.copy()
    pixels = img.load()
    suite_cnt = Counter()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            if sum(pixels[i, j]) >= 255 * 2:
                pixels[i, j] = (255, 255, 255)
            else:
                if pixels[i, j][0] > 255 * 2 // 3:
                    suite_cnt["R"] += 1
                elif pixels[i, j][1] > 255 * 2// 5:
                    suite_cnt["G"] += 1
                else:
                    suite_cnt["B"] += 1
                pixels[i, j] = (0, 0, 0)
    colors = set(col for col, cnt in suite_cnt.items() if cnt > 20)
    if "R" in colors and "G" in colors:
        return "S",img
    return max(suite_cnt.items(), key=lambda x: x[1])[0], img


def load_number_predictor():
    output_classes = sorted([p.name for p in Path("./train").iterdir()])
    model, input_size = initialize_model(
        MODEL_NAME,
        num_classes=len(output_classes),
        feature_extract=True,
        use_pretrained=True,
    )
    model.load_state_dict(torch.load("model.torch"))
    model.eval()
    IMG_MEAN = (0.485, 0.456, 0.406)
    IMG_STD = (0.229, 0.224, 0.225)
    transform = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD),
        ]
    )

    def predict(x):
        with torch.no_grad():
            x = transform(x)
            batch = x.unsqueeze(0)
            out = model(batch)
            pred = int(torch.argmax(out).item())
            return output_classes[pred]

    return predict


def load_state(file):
    predict_number = load_number_predictor()
    res = []
    for img, i, j in extract_imgs(file):
        suite, bw_img = predict_color(img)
        num = predict_number(bw_img)
        if num in SUITE_MAP:
            suite, num = SUITE_MAP[num]
        if suite == "S":
            num = "1"
        img.save(f"imgs/{i}x{j}x{suite}x{num}.png")
        res.append(f"{suite}{num}")
    return res


if __name__ == "__main__":
    res = load_state("./monitor-1.png")
    print(res)
