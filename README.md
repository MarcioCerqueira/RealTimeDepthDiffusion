# Live User-Guided Depth Map Estimation for Single Images

by Márcio C. F. Macedo and Antônio L. Apolinário Jr.

## Introduction

This is a C++ application for user-guided, real-time depth map estimation in single images. Technical details are provided in our [paper](https://doi.org/10.1007/s11554-020-01055-x) accepted in the Journal on Real-Time Image Processing.

The provided source code was tested using the following libraries:
* Eigen 3.3.9;
* OpenCV 4.5.1;
* CUDA 11.2;

The application receives as input the following console arguments:
* ```-i image.extension```: to provide an input image for the application;
* ```-a image.extension```: to provide an initial annotation for the input image;
* ```--live```: to enable live depth annotation;

Once the application is started, one can interact with the depth map estimation using the following keys:
* Press '0' to annotate the "Edited Image" with depth 0;
* Press '1' to annotate the "Edited Image" with depth 64;
* Press '2' to annotate the "Edited Image" with depth 128;
* Press '3' to annotate the "Edited Image" with depth 192;
* Press '4' to annotate the "Edited Image" with depth 255;
* Press '+' to increase the scribble radius;
* Press '-' to decrease the scribble radius;
* Press 't' or 'T' to print the processing time demanded by one frame into the console;
* Press 's' or 'S' to save the estimated depth map, the annotated image and the artistic depth-based effect;
* Press 'd' or 'D' to start the depth map estimation process (this process is automatic if --live is enabled);
* Press 'g' or 'G' to visualize the desaturation effect (the output image is automatically updated if --live is enabled);
* Press 'h' or 'H' to visualize the haze effect (the output image is automatically updated if --live is enabled);
* Press 'b' or 'B' to visualize the refocus effect (the output image is automatically updated if --live is enabled);

## Citation

The provided source codes are in public domain and can be downloaded for free. If this work is useful for your research, please consider citing:

  ```shell
  @article{Macedo2021,
  author={Macedo, M{\'a}rcio C. F. and Apolin{\'a}rio, Ant{\^o}nio L.},
  title={Live User-Guided Depth Map Estimation for Single Images},
  journal={Journal of Real-Time Image Processing},
  year={2021},
  month={Jan},
  day={13},
  issn={1861-8219},
  doi={10.1007/s11554-020-01055-x},
  }
  ```
