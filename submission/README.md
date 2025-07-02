# Trackerless 3D Freehand Ultrasound Reconstruction Challenge 2025 (TUS-REC2025)
<!-- ## About -->

[Submission Requirement](#submission-requirement) |
[Data Structure](#data-structure) |
[Instructions for Docker](#instructions-for-docker) |
[Back to Main Page](../README.md)


## Submission requirement
* We have provided an example docker image [here](#instructions-for-docker), which can predict DDFs on the validation/test dataset. The source code is also available in [`submission`](https://github.com/QiLi111/TUS-REC2025-Challenge_baseline/tree/main/submission) folder.
* The participants are expected to replace the content of [`predict_ddfs`](https://github.com/QiLi111/TUS-REC2025-Challenge_baseline/blob/main/submission/predict_ddfs.py) function with their own algorithm, which is used in [test.py](https://github.com/QiLi111/TUS-REC2025-Challenge_baseline/blob/a818cdb708049b6a2209b7dbde6759ef1c8af0e8/submission/test.py#L39). The function is expected to take one entire scan as input, and output four DDFs. There is no requirement on how the algorithm is designed internally, for example, whether it is learning-based method; frame-, sequence- or scan-based processing; or, rigid-, affine- or nonrigid transformation assumptions.  
* The requirement of the `predict_ddfs` function is described below:
  * Input: 
    * `frames`: All frames in the scan; numpy array with a shape of [N,480,640], where N is the number of frames in this scan.
    * `landmark`: Location of 100 landmarks in the scan; numpy array with a shape of [100,3]. Each row denotes one landmark and the three columns denote the frame index (starting from 0) and the 2d-coordinates of landmarks in the image coordinate system (starting from 1, to maintain consistency with the calibration process). For example, a row like [10,200,100] indicates that there is a landmark in the 10th frame, located at the coordinates [200,100].
    * `data_path_calib`: Path to calibration matrix.
    * `device`: Device to run the model on, provided in [this line](https://github.com/QiLi111/TUS-REC2025-Challenge_baseline/blob/a818cdb708049b6a2209b7dbde6759ef1c8af0e8/submission/test.py#L26).
  * Output:  
     * `GP`: Global displacement vectors for all pixels. DDF from the current frame to the first frame, in mm. The first frame is regarded as the reference frame. The DDF should be in numpy array format with a shape of [N-1,3,307200] where N-1 is the number of frames in that scan (excluding the first frame), "3" denotes “x”, “y”, and “z” axes, respectively, and 307200 is the number of all pixels in a frame. The order of the flattened 307200 pixels can be found in function [`reference_image_points`](https://github.com/QiLi111/TUS-REC2025-Challenge_baseline/blob/a818cdb708049b6a2209b7dbde6759ef1c8af0e8/submission/utils/plot_functions.py#L6).
     * `GL`: Global displacement vectors for landmarks, in mm. The DDF should be in numpy array format with a shape of [3,100], where 100 is the number of landmarks in a scan.
     * `LP`: Local displacement vectors for all pixels. DDF from current frame to the previous frame, in mm. The previous frame is regarded as the reference frame. The DDF should be in numpy array format with a shape of [N-1,3,307200], where N-1 is the number of frames in that scan (excluding the first frame), "3" denotes “x”, “y”, and “z” axes, respectively, and 307200 is the number of all pixels in a frame. The order of the flattened 307200 pixels can be found in function [`reference_image_points`](https://github.com/QiLi111/TUS-REC2025-Challenge_baseline/blob/a818cdb708049b6a2209b7dbde6759ef1c8af0e8/submission/utils/plot_functions.py#L6).
     * `LL`: Local displacement vectors for landmarks, in mm. The DDF should be in numpy array format with a shape of [3,100], where 100 is the number of landmarks in a scan.
     
        
> [!NOTE]  
> * If you are not sure about data dimensions, coordinate system or transformation direction, etc., please refer to the [example code](https://github.com/QiLi111/TUS-REC2025-Challenge_baseline/blob/main/submission/baseline_model/Prediction.py) in `baseline_model` folder. 
>* We have provided two functions, which can generate four DDFs from global and local transformations, in [`Transf2DDFs.py`](https://github.com/QiLi111/TUS-REC2025-Challenge_baseline/blob/main/submission/utils/Transf2DDFs.py).
> * Only modify the implementation of the `predict_ddfs` function. It’s okay to add files but please do not change existing files other than `baseline_model` folder.
> *  The order of the four DDFs and the order of 307200 pixels cannot be changed and they must all be numpy arrays. Please ensure your prediction does not have null values. Otherwise, the final score could not be generated.  
> * Your model is expected to run on a single GPU, with GPU memory usage not exceeding 32GB when running docker.
> * Participants are required to dockerize their trained network/algorithm/method and submit them via a file-sharing link (e.g., OneDrive, Dropbox) to the organizers via this <a href="https://forms.office.com/e/dj1g5TKyaj" target="_blank">form</a>.
> * Participants are allowed to make multiple distinct submissions (but must ensure they are not merely simple variations in hyperparameter values), and the best result will be selected for competing. The number of submissions for each team is limited to 5 to preserve variations in hyperparameters.
> * The test set in the TUS-REC2025 Challenge includes only the rotation scans, but you may use the datasets from the TUS-REC2024 Challenge as extra training data.

## Data structure
We have provided validation data [here](https://doi.org/10.5281/zenodo.15699958), which has the same structure as the test data. A successful run on the validation set will ensure that the code will run without problems on the holdout test set. The data folder structure is as follows. Details can be found in the [zenodo page](https://doi.org/10.5281/zenodo.15699958). The validation set is different from the training set in that: 1) the image and transformation for each scan are stored separately in two folders; 2) the added file `dataset_keys.h5` contains the paths to all scans in the dataset.

```bash
Freehand_US_data_val_2025/ 
    │
    ├── frames/
    │   ├── 050/
    │       ├── RH_rotation.h5 # US frames in rotating scan of right forearm, subject 050
    │       └── LH_rotation.h5 # US frames in rotating scan of left forearm, subject 050
    │   
    │   ├── 051/
    │       ├── RH_rotation.h5 # US frames in rotating scan of right forearm, subject 051
    │       └── LH_rotation.h5 # US frames in rotating scan of left forearm, subject 051
    │   
    │   ├── ...
    │
    │
    ├── transfs/
    │   ├── 050/
    │       ├── RH_rotation.h5 # transformations (from tracker tool space to optical camera space) in rotating scan of right forearm, subject 050
    │       └── LH_rotation.h5 # transformations (from tracker tool space to optical camera space) in rotating scan of left forearm, subject 050
    │   
    │   ├── 051/
    │       ├── RH_rotation.h5 # transformations (from tracker tool space to optical camera space) in rotating scan of right forearm, subject 051
    │       └── LH_rotation.h5 # transformations (from tracker tool space to optical camera space) in rotating scan of left forearm, subject 051
    │   
    │   ├── ...
    │
    │
    ├── landmarks/
    │   ├── landmark_050.h5 # landmark coordinates in scans of subject 050
    │   ├── landmark_051.h5 # landmark coordinates in scans of subject 051
    │   ├── ...
    │
    ├── calib_matrix.csv # calibration matrix
    └── dataset_keys.h5 # contains paths to all scans in the dataset


```


<!-- 
* The data structure of validation data set is explained as below:
  * Folder `frames`: contains three folders (one subject per folder), each with 24 scans. Each .h5 file corresponds to one scan, storing image of each frame within this scan. Key-value pair and name of each .h5 file are explained below. 
    * "frames" - All frames in the scan; with a shape of [N,H,W], where N refers to the number of frames in the scan, H and W denote the height and width of a frame. 
    * Notations in the name of each .h5 file: “RH”: right arm; “LH”: left arm; “Per”: perpendicular; “Par”: parallel; “L”: straight line shape; “C”: C shape; “S”: S shape; “DtP”: distal-to-proximal direction; “PtD”: proximal-to-distal direction; For example, “RH_Per_L_DtP.h5” denotes a scan on the right forearm, with ultrasound probe perpendicular of the forearm sweeping along straight line, in distal-to-proximal direction.
  * Folder `transfs`: contains three folders (one subject per folder), each with 24 scans. Each .h5 file corresponds to one scan, storing transformation of each frame within this scan. Key-value pair and name of each .h5 file are explained below. 
     * "tforms" - All transformations in the scan; with a shape of [N,4,4], where N is the number of frames in the scan, and the transformation matrix denotes the transformation from tracker tool space to camera space. 
    * Notations in the name of each .h5 file is the same as in folder `frames`.
  * Folder `landmark`: contains three .h5 files. Each corresponds to one subject, storing coordinates of landmarks for 24 scans of this subject. For each scan, the coordinates are stored in numpy array with a shape of 20×3. The first column is the index of frame; the second and third columns denote the coordinates of landmarks in the image coordinate system.
  * `calib_matrix.csv`: The calibration matrix was obtained using a pinhead-based method. The "scaling_from_pixel_to_mm" and "spatial_calibration_from_image_coordinate_system_to_tracking_tool_coordinate_system" are provided in the “calib_matrix.csv”.
  * `dataset_keys.h5`: stores all the scan name information. Keys in “dataset_keys.h5” denotes all the available scans in test set, in a format of “sub%03d__%s” where sub%03d denotes which folder, and %s denotes the scan name. For example, “sub050__LH_Par_C_DtP” means the scan in folder “050”, with file name of “LH_Par_C_DtP.h5” -->

## Instructions for docker
This section contains instructions to build and run docker image. 
Note: sudo or docker group permissions may be needed to run the following commands.

### Prerequisites
* [Docker Engine](https://docs.docker.com/engine/install/)
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#)
<!-- 
If you are using GPU, NVIDIA Container Toolkit may need to be installed. You can refer to the [official website](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) or follow the steps below.

#### 1. Configure the production repository.
```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
#### 2. Update the packages list from the repository.
```
sudo apt-get update
```
#### 3. Install the NVIDIA Container Toolkit packages.
```
sudo apt-get install -y nvidia-container-toolkit
```
#### 4. Configure the container runtime by using the `nvidia-ctk` command.
```
sudo nvidia-ctk runtime configure --runtime=docker
```
#### 5. Restart the Docker daemon.
```
sudo systemctl restart docker
``` -->

### Instructions to build and run docker
In case you are not familiar with docker, we describe below the whole process of creating and running docker image to generate DDFs, using our baseline model.
#### 1. Clone the repository.
```
git clone https://github.com/QiLi111/TUS-REC2025-Challenge_baseline.git
```

#### 2. Navigate to the `submission` directory.
```
cd TUS-REC2025-Challenge_baseline/submission
```
<!-- Note: If you want to see the plotted trajectories, please uncomments [this](https://github.com/QiLi111/tus-rec-challenge_baseline/blob/5eec6014a7de2b652cdcf7333c58e00f01348560/submission/test.py#L44) line. -->

#### 3. Prepare Dockerfile.
You can start from our example [Dockerfile](https://github.com/QiLi111/TUS-REC2025-Challenge_baseline/blob/main/submission/Dockerfile). To use your own environment, you just need to modify the [`requirements.txt`](https://github.com/QiLi111/TUS-REC2025-Challenge_baseline/blob/main/submission/requirements.txt) to specify the dependencies.

#### 4. Build docker image using Dockerfile. 

```
docker build -t tus-rec2025:v1 .
```
The `-t tus-rec2025:v1` option specifies the name and tag of the image. The single dot (`.`) at the end of the command sets the build context to the current directory. You may refer to the [official website](https://docs.docker.com/build/building/packaging/#building) for detailed information. After building the docker image, you can run `docker images` to check a list of all local Docker images. 

#### 5. Download Validation Dataset `Freehand_US_data_val_2025.zip` [here](https://doi.org/10.5281/zenodo.15699958) into `submission` folder. (You may need to install zenodo_get)
```
pip3 install zenodo_get
zenodo_get 15699959
```

#### 6. Unzip.
Unzip Freehand_US_data_val_2025.zip into `data` folder.
```
unzip Freehand_US_data_val_2025.zip -d ./data
```

#### 7. Run docker.
* If your Docker Engine version is above 23, relative path is supported when mounting external folders into docker container.

  ```
  docker container run --gpus all -it -v ./data:/home/TUS-REC2025/data -v ./results:/home/TUS-REC2025/results --rm tus-rec2025:v1
  ```
* If your Docker Engine version is below 23, absolute path is compulsory.

  ```
  docker container run --gpus all -it -v /path/to/your/data/directory/on/the/host/machine:/home/TUS-REC2025/data -v /path/to/your/results/directory/on/the/host/machine:/home/TUS-REC2025/results --rm tus-rec2025:v1
  ```

  The `--gpu` option specifies which gpu to use, and the `-v` option specifies the mounted directories (`data` and `results`) on the host machine, the `-it` option opens interactive window to receive standard output and the `--rm` option means the container will be deleted once exit (more instruction [here](https://docs.docker.com/reference/cli/docker/container/run/#volume)). 


<!-- You can check the status of all the running containers using command `docker ps`. -->
  
  After successfully running the docker, you will see the `results` folder. 

#### 8. Export docker image.
Export image `tus-rec2025:v1` to `tus-rec2025.tar`
```
docker save -o tus-rec2025.tar tus-rec2025:v1
```

You can also export the compressed Docker image using command below (more instruction [here](https://docs.docker.com/reference/cli/docker/image/save/)).
```
docker save tus-rec2025:v1 | gzip > tus-rec2025.tar.gz
```

