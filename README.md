# Subject Mask Generator
This repository contains a script to generate grayscale mask images suitable
for use in diffusion model fine-tuning for human subjects.

It supports specifying different weights for face, hair, body and background.
It also supports automatic selection of the correct subject when multiple persons exist in the same input image, using face recognition.

## Installation
1. Clone this repository
2. Create a python virtual environment
2. Install PyTorch
2. Install dependencies from `requirements.txt`

## Running
Example:
``python main.py --input-dir /path/to/image/folder --output-dir /path/to/destination/folder --target-ref /path/to/subject/face``

For documentation on all the script parameters please take a look inside the script file, at the bottom.
