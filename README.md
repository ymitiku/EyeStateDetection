# EyeStateDetection
Demo project to detect if eye is open or closed

## How to run demo
### Image demo

```
python -m demo --process image --path /path/to-image --json_path /path/to-model/json/file --weights /path/to/weights of the model
```
Where 
* ```--process``` demo type. It can be either ```image``` for image demo, ```webcam``` for webcam demo
or ```video``` for video demo. 
* ```--path``` is full path to image. 
* ```--json_path``` path to model's json file
* ```--weights``` path to weights of model(h5 file)

### video demo
```
python -m demo  --process video --path /path-to-video/video-file --json_path /path/to-model/json/file --weights /path/to/weights of the model
```
Where ```--path``` is full path to video.
### web demo
```
python -m demo --process webcam --json_path /path/to-model/json/file --weights /path/to/weights of the model
```

### Dependancies

* tensorflow >= 1.0
* keras >= 2.0
* opencv >= 3.0
* dlib 
* numpy

* [shape_predictor_68_face_landmarks.dat][sp]

#### N.B

* **opencv should be compiled with ffmpeg support.**
* **Conda virtual environment can be created using the following command.**

 ```
 conda env create -f requirements.yml -n emopy_2
 ```
* **shape_predictor should be inside root directory of this project. Shape predictor can be downloaded to project using the following script.**
```
cd /path-to-project
wget "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```

 [sp]: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
