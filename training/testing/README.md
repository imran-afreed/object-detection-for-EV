
### TESTING:

1) The trained weights file shoud be added here 
2) The custom configuration file yolov3-custom.cfg created while training has to be added here
3) detect.py takes pictures from directory specified in line 16 in its code
```python
#images_path = glob.glob(r"address of image directory/*.jpg")
images_path = glob.glob(r"/home/afreed/Desktop/CNN/cones/cone_detection/*.jpg")
```
4) Run ``` python detect.py ``` to test

