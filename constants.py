PARAMS_FILE = "params_file.txt"

CLASSES = ["abort", "command", "correct", "incorrect", "prompt", "reward", "visual"]

img_h = 480
img_w = 640
img_size = 299
c_size = 64
aud_h = 128
aud_w = 8

img_dtype = {
    "name": "img",
    "img_h": img_h,
    "img_w": img_w,
    "num_c": 3,
    "cmp_h": img_size,
    "cmp_w": img_size
}

img_resize_dtype = {
    "name": "img_resize",
    "img_h": 64,
    "img_w": 64,
    "num_c": 3,
    "cmp_h": 64,
    "cmp_w": 64
}

grs_dtype = {
    "name": "grs",
    "img_h": img_h,
    "img_w": img_w,
    "num_c": 1,
    "cmp_h": img_size,
    "cmp_w": img_size
}

grs_resize_dtype = {
    "name": "grs_resize",
    "img_h": 64,
    "img_w": 64,
    "num_c": 1,
    "cmp_h": 64,
    "cmp_w": 64
}

pnt_dtype = {
    "name": "pnt",
    "img_h": img_h,
    "img_w": img_w,
    "num_c": 1,
    "cmp_h": c_size,
    "cmp_w": c_size
}

aud_dtype = {
    "name": "aud",
    "img_h": img_h,
    "img_w": img_w,
    "num_c": 1,
    "cmp_h": aud_h,
    "cmp_w": aud_w
}
