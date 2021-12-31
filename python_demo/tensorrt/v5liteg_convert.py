# YOLOv5 experimental modules
import torch
from utils.google_utils import attempt_download


if __name__ == '__main__':
    load = 'weights/repyolov5s.pt'
    save = 'weights/repyolov5s-opt.pt'
    input_size = 640
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for w in load if isinstance(load, list) else [load]:
        attempt_download(w)
        ckpt = torch.load(w, map_location=None)  # load

    torch.save(ckpt, save)

    print(f'Done. Befrom weights:({load})')
    print(f'Done. Befrom weights:({save})')