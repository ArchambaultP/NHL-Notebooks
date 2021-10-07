from PIL import Image
from pathlib import Path
from os.path import dirname, abspath

def get_icerink():
    p = Path(dirname(abspath(__file__))).parent.parent
    img_file = p / "figures/nhl_rink.png"
    return Image.open(img_file)
