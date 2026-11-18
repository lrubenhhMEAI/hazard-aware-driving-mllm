# Data

This project uses the **DRAMA** dataset (Malla et al., 2023). It is distributed under its own licence and is **not included** in this repository.

1. Request access from the dataset authors (Honda Research Institute).
2. Place the files like this:
```
data/drama/
├── integrated_output_v2.json   # annotations
└── combined/                   # extracted frames (clip folders with frame_XXXXXX.png)
```
Or point to another location with the `DRAMA_ANNOTATIONS` and `DRAMA_FRAMES` environment variables.
