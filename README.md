# Censorship... with style! ™

### An extension for [SD.Next](https://github.com/vladmandic/automatic)

## Features

- Detect geneder and any body part  
  e.g. *female face, belly, feet*
- Exposed or unexposed:  
  e.g. *breast or breast-bare*
- Add information to image metadata
  e.g. *"NudeNet: female-face:0.86; belly:0.54"*, *NSFW: True*  
- Censor as desired (or not):
  - blur *(adjust block size for effect)*
  - pixelate *(adjust block size for effect)*
  - cover with pasty (overlay image) :)  
    *note: RGBA image is recommended for overlays*  
- Use as extension from UI or via CLI  
  `python nudenet.py --help`  
- Adjustable sensitivity
- Can be used for **txt2img**, **img2img** or **process**  
- FAST!  
  Uses `CV2` and `ONNX` backend and typically executes in <0.1sec  

## Settings

Should be self-explanatory...

![settings](settings.png)

## Examples

![example-pasty](example-pasty.jpg)
![example-pixelate](example-pixelate.jpg)
