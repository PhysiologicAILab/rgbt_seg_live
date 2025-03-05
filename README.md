# TCC RGB+T Fusion data collection & visualisation software
## project code: rgbt_seg_live (public release date: 5 March 2025)

This repo was developed as a part of TCC project and offers support for following:

- Capturing RGB and/or Thermal video frames
- Real-time display of captured RGB and thermal frames.
- Psuedocolor visualization of thermal frames for intuitive inspection of garment damages.
- Automated detection of damaged regions using semantic segmentation model trained using our acquired dataset.


## User Interface for TCC RGB+T Fusion Data Collection & Visualisation Software

<img src="images/image.png" alt="User interface for TCC RGB+T Fusion data collection & visualisation software" width=720/>

## Running the Interface

``` bash
python -m rgbt_main --savepath "<path to save the recorded frames>"
```


### Acknowledgement: This project was funded by the Engineering and Physical Sciences Research Council (EP/V011766/1) for the UK Research and Innovation (UKRI) Interdisciplinary Circular Economy Centre for Textiles: Circular Bioeconomy for Textile Materials.
