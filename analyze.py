from pathlib import Path
from PIL import Image, ImageDraw
from collections import Counter
import platform
import subprocess

if platform.python_implementation() == "CPython":
    from train import initialize_model, MODEL_NAME

    print("CPYTHON")
    from torchvision import transforms
    import torch

START = (52, 287)
COLUMN_OFFSET = 152
ROW_OFFSET = 31
WIDTH = 20
SUITE_MAP = {
    "box": ("B", 0),
    "chin": ("G", 0),
    "y": ("R", 0),
    "special": ("S", 1),
}
GAME_BOX = (358, 108, 1634, 958)
GAME_BOX = (358, 91, 1634, 958)


def extract_imgs(file):
    cropped_imgs = []
    with Image.open(file) as img:
        img = img.crop(GAME_BOX)
        draw = ImageDraw.Draw(img)
        for i in range(5):
            for j in range(8):
                box = (
                    START[0] + j * COLUMN_OFFSET,
                    START[1] + ROW_OFFSET * i,
                    START[0] + j * COLUMN_OFFSET + WIDTH,
                    START[1] + ROW_OFFSET * (i + 1) - 7,
                )
                cropped_imgs.append((img.crop(box), i, j))
                draw.line(box, fill=128)
        top_start_x = START[0] + 5 * COLUMN_OFFSET
        top_start_y = START[1] - 13 * WIDTH - 5
        for j in range(3):
            box = (
                top_start_x + j * COLUMN_OFFSET,
                top_start_y,
                top_start_x + j * COLUMN_OFFSET + WIDTH,
                top_start_y + ROW_OFFSET - 7,
            )
            cropped_imgs.append((img.crop(box), -1, j))
            draw.line(box, fill=128)
        # special
        special_offset = COLUMN_OFFSET + 40
        box = (
            top_start_x - special_offset,
            top_start_y,
            top_start_x - special_offset + WIDTH,
            top_start_y + ROW_OFFSET - 7,
        )
        cropped_imgs.append((img.crop(box), -1, -1))
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
                elif pixels[i, j][1] > 255 // 4:
                    suite_cnt["G"] += 1
                else:
                    suite_cnt["B"] += 1
                pixels[i, j] = (0, 0, 0)
    if suite_cnt["G"] > 200:
        return "E", img
    colors = set(col for col, cnt in suite_cnt.items() if cnt > 20)
    if "R" in colors and "G" in colors:
        return "S", img
    return max(suite_cnt.items(), key=lambda x: x[1], default=("S", 0))[0], img


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
    cards = []
    for img, i, j in extract_imgs(file):
        suite, bw_img = predict_color(img)
        num = predict_number(bw_img)
        if num in SUITE_MAP:
            suite, num = SUITE_MAP[num]
        if suite == "S":
            num = "1"
        # img.save(f"imgs/{i}x{j}x{suite}x{num}.png")
        cards.append(f"{suite}{num}")
    res = []
    specials = []
    order = [c for c, _ in cards[-4:-1] if c != "E"]
    for c in cards[-4:]:
        if c[0] == "E":
            continue
        specials.append(c)
        if c[0] == "S":
            continue
        for num in reversed(range(1, int(c[1]))):
            specials.append(f"{c[0]}{num}")
    for _ in range(8 * 5):
        card = cards.pop(0)
        if card in res and int(card[1]) > 0 and specials:
            res.append(specials.pop(0))
        else:
            res.append(card)
    return res, order


def load_state_indirect(screenshot):
    out = subprocess.check_output(
        [
            r"C:\Users\dwestreicher\AppData\Local\Microsoft\WindowsApps\python.exe",
            "analyze.py",
        ]
    ).decode()
    lines = [l for l in out.split("\n") if l]
    return eval(lines[-1])


if __name__ == "__main__":
    res = load_state("./monitor-1.png")
    print(res)
