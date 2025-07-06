# Trackerless 3D Freehand Ultrasound Reconstruction Challenge 2025 (TUS-REC2025)
<!-- ## About -->

<a href="https://github-pages.ucl.ac.uk/tus-rec-challenge/" target="_blank">Website</a> |
<a href="https://zenodo.org/records/15224704" target="_blank">Train Dataset</a> |
[Training Code Usage Guide](#training-code-usage-guide) |
<a href="https://doi.org/10.5281/zenodo.15699958" target="_blank">Validation Dataset</a> |
<a href="/submission/README.md" target="_blank">Submission Requirement and Example Docker</a> |
[Data Usage Policy](#data-usage-policy)
<!-- [Data Usage Policy](#data-usage-policy) -->

<div align=center>
  <a target="_blank"><img style="padding: 10px;" src="img/logo.png" width=200px></a>
</div >

**Welcome to the Trackerless 3D Freehand Ultrasound Reconstruction Challenge 2025 (TUS-REC2025)！**   

The TUS-REC2025 Challenge is a part of the 28th International Conference on Medical Image Computing and Computer Assisted Intervention (<a href="https://conferences.miccai.org/2025/en/" target="_blank">MICCAI 2025</a>), held in conjunction with the 6th ASMUS workshop, September 23rd 2025 in Daejeon, Republic of Korea. The challenge is supported by the MICCAI Special Interest Group in Medical Ultrasound (<a href="https://miccai.org/index.php/special-interest-groups/sig/" target="_blank">SIGMUS</a>) and will be presented at its international workshop <a href="https://miccai-ultrasound.github.io/#/asmus25" target="_blank">ASMUS 2025</a>.

**You can jump to [Training Code Usage Guide](#training-code-usage-guide) if you have read the introduction in <a href="https://github-pages.ucl.ac.uk/tus-rec-challenge/" target="_blank"> challenge website.</a>**

## Background
Reconstructing 2D Ultrasound (US) images into a 3D volume enables 3D representations of anatomy to be generated which are beneficial to a wide range of downstream tasks such as quantitative biometric measurement, multimodal registration, and 3D visualisation. This application is challenging due to 1) inherent accumulated error - frame-to-frame transformation error will be accumulated through time when reconstructing long sequence of US frames, and 2) a lack of publicly-accessible data with synchronised spatial location, often obtained from tracking devices, for benchmarking the performance and for training learning-based methods.

TUS-REC2025 presents a different scanning protocol, in addition to the previous TUS-REC2024 non-rotation-based protocols. The new scans include more diverse probe movement such as rotating and tilting at various angles. With 3D reconstruction as the challenge task, TUS-REC2025 aims to 1) benchmark the model performance on the new rotating data, and 2) validate the model generalisation ability among different scan protocols. The outcome of the challenge includes 1) providing in open access the new US dataset with accurate positional information; 2) establishing the first benchmark for 3D US reconstruction for rotating scans, suitable for modern learning-based data-driven approaches.


## Freehand US Reconstruction

The aim of Freehand US reconstruction is to estimate the transformation between any pair of US frames in an US scan without any external tracker, and thus reconstruct 2D US images into a 3D volume (see [Fig. 1](#figure1)).

<div align=center>
  <a 
  target="_blank"><img 
  style="padding: 10px;" 
  src="img/rec.png" 
  width=300px
  id="figure1">
  
</a>
</div >

<div align=center>
Fig. 1. An illustration of freehand US reconstruction.
</div >
<!-- <p align="center">
  <img src="img2025/rec.png" />
</p> -->
<!-- <figure>
  <img src="img2025/rec.png" alt="An example workflow of freehand US reconstruction" width="400"/>
  <figcaption>Figure 1: An example workflow of freehand US reconstruction.</figcaption>
</figure> -->


For an US scan $\mathcal{S}$, image sequences comprising $M$ 2D frames can be sampled as $s_m=\{I_m\}, m=1,2,...,M$, where $s_m \subseteq {\mathcal{S}}$ and $m$ represents consecutively increasing time-steps at which the frames are acquired. [Fig. 2](#figure2) shows the relationship among three coordinate systems: the image coordinate system, the tracker tool coordinate system, and the camera coordinate system.

<div align=center>
  <a 
  target="_blank"><img 
  style="padding: 10px;" 
  src="img/coordinate_system.png" 
  width=300px
  id="figure2">
  
</a>

</div >
<div align=center>
Fig. 2. The relationship among three coordinate systems: the image coordinate system, the tracker tool coordinate system, and the camera coordinate system.
</div>

The rigid transformation from the $i^{th}$ frame to the $j^{th}$ frame (in mm), $T_{j\leftarrow i}$, can be obtained using Eq. 1, where $T_{j\leftarrow i}^{tool}$ denotes the transformation between $i^{th}$ tacker tool to the $j^{th}$ track tool and $T_{rotation}$ represents spatial calibration from image coordinate system (in mm) to tracking tool coordinate system.

<!-- <a id="transformation"></a> -->
$$
T_{j\leftarrow i}= T_{rotation}^{-1} \cdot T_{j\leftarrow i}^{tool} \cdot T_{rotation}\quad (1)
$$
<!-- , 1 \leq i<j \leq M  -->

In general, prior studies have formulated freehand US reconstruction as the estimation of the transformation between two frames in an US sequence. This estimation relies on a function $f$, which serves as the core of freehand US reconstruction, as expressed in Eq. 2: 

<!-- <a id="freehandUS"></a>

\begin{equation} -->
$$
T_{j\leftarrow i} \approx f(I_i, I_j) \quad (2)
$$
<!-- \end{equation} -->


Typically, adjacent frames are used in Eq. 2. The transformation from $i^{th}$ frame to the first frame $T_{1\leftarrow i}$ can be computed by recursively multiplying the previously estimated relative transformations, as shown in Eq. 3:

<!-- <a id="chain-multiplying"></a>
\begin{equation} -->
$$
T_{1\leftarrow i}= T_{1\leftarrow 2} \cdot T_{2\leftarrow 3}  \cdots  T_{i-1\leftarrow i} \quad (3)
$$
<!-- \end{equation} -->


Moreover, Eq. 3 demonstrates that estimation errors can propagate and accumulate throughout the chain, ultimately resulting in trajectory drift.

Reconstructing the 3D US volume and the trajectory of the US frames requires determining the position of each frame. 
The first frame is chosen as the reference. As a result, only the relative transformations with respect to the first frame are needed.
For any pixel $x$ in $i^{th}$ frame with coordinates $p_x$ in image coordinate system (in pixel) of frame $i$, the coordinates in image coordinate system (in mm) of frame 1, $P_x$, can be obtained using Eq. 4.

<!-- <a id="coordinate"></a> -->

<!-- \begin{equation} -->
$$
P_x = T_{1\leftarrow i} \cdot T_{scale} \cdot p_x  \quad (4)
$$
<!-- \end{equation} -->

where $T_{scale}$ denotes the scaling from pixel to mm.
<!-- where $T_i$ denotes the transformation from $i^{th}$ frame to the first frame. -->


## Task Description
The aim of this task is to reconstruct 2D US images into a 3D volume. The algorithm is expected to take the entire scan as input and output two different sets of transformation-representing displacement vectors as results, a set of displacement vectors on individual pixels and a set of displacement vectors on provided landmarks. There is no requirement on how the algorithm is designed internally, for example, whether it is learning-based method; frame-, sequence- or scan-based processing; or, rigid-, affine- or nonrigid transformation assumptions. Details are explained further in <a href="https://github-pages.ucl.ac.uk/tus-rec-challenge/assessment.html" target="_blank">Assessment</a>.

Participant teams are expected to make use of the sequential data and potentially make knowledge transfer from US data with other scanning protocols, for example the dataset released in TUS-REC2024. The participant teams are expected to take US scan as input and output two sets of pixel displacement vectors, indicating the transformation to reference frame. The evaluation process will take the generated displacement vectors from their dockerized models, and produce the final accuracy score to represent the reconstruction performance, at local and global levels, representing different clinical application of the reconstruction methods.

For detailed information, please see <a href="https://github.com/QiLi111/TUS-REC2025-Challenge_baseline/blob/main/generate_DDF.py" target="_blank">generate_DDF.py</a> and <a href="https://github-pages.ucl.ac.uk/tus-rec-challenge/assessment.html#metrics" target="_blank">Metrics</a> for an example of generating four DDFs.

### Difference between TUS-REC2025 and TUS-REC2024

From the results of TUS-REC2024, we observed that the reconstruction performance is dependent on scan protocol. In TUS-REC2025, we want to investigate the reconstruction performance on scans with a new rotating scanning protocol, with which the reconstruction performance may be further improved owing to its dense sampling of the area to be reconstructed. Compared with TUS-REC2024, TUS-REC2025 provides more data with new scanning protocol, and the previous released larger data with non-rotating scanning protocols is open to use. The new challenge aims to 1) benchmark the model performance on relatively small rotating data and 2) benchmark the model generalisation ability among different scanning protocols.

## Dataset

The data in this challenge is acquired from both left and right forearms of 85 volunteers, acquired at University College London, London, U.K, with a racial-, gender-, age-diverse subject cohort. [Fig. 3](#figure3) shows the equipment setting during acquisition. No specific exclusion criteria as long as the participants do not have allergies or skin conditions which may be exacerbated by US gel. All scanned forearms are in good health. The data is randomly split into train, validation, and test sets of 50, 3, and 32 subjects (100, 6, 64 scans; ~164k, ~9k, ~90k frames), respectively.

<div align=center>
  <a 
  target="_blank"><img 
  style="padding: 10px;" 
  src="img/data_acqusition.png" 
  width=300px
  id="figure3">
  
</a>

</div >
<div align=center>
Fig. 3. Freehand US data acquisition system.
</div>

### Images

The 2D US images were acquired using an Ultrasonix machine (BK, Europe) with a curvilinear probe (4DC7-3/40). The acquired US frames were recorded at 20 fps, with an image size of 480×640, without speckle reduction. The frequency was set at 6MHz with a dynamic range of 83 dB, an overall gain of 48% and a depth of 9 cm. Both left and right forearms of volunteers were scanned. For each forearm, the US probe was positioned near the elbow and moved around the fixed contact point. It was first fanned side-to-side along the short axis of the skin-probe interface and then rocked along the long axis in a similar manner. Afterwards, the probe was rotated about 90 degrees, and the fanning and rocking motions were repeated. The dataset contains 170 scans in total, 2 scans associated with each subject, around 1500 frames for each scan.

### Labels / Transformations

The position information recorded by the optical tracker (NDI Polaris Vicra, Northern Digital Inc., Canada) will be provided along with the images, which indicates the position of the US probe for each frame in the camera coordinate system, described as homogeneous transformation matrix with respect to reference frame. A calibration matrix will also be provided, denoting the transformation between US image coordinate system and tracker tool coordinate system while these data were acquired. The data is provided temporally calibrated, aligning the timestamps for both transformations from the optical tracker and ultrasound frames from US machine.

An illustration of a scan is shown in [Fig. 4](#figure4).

<div align=center>
  <a 
  target="_blank"><img 
  style="padding: 10px;" 
  src="img/example_scan.gif" 
  width=300px
  id="figure4">
  
</a>

</div >
<div align=center>
Fig. 4. An illustration of a rotation scan. (The 3D arm stl model is from <a href="https://www.printables.com/model/349380-hand-and-forearm-natural-position-scan/files" target="_blank">here</a>.)
</div>

### Train Data Structure: 
```bash

Freehand_US_data_train_2025/ 
    │
    ├── frames_transfs/
    │   ├── 000/
    │       ├── RH_rotation.h5 # US frames and associated transformations (from tracker tool space to optical camera space) in rotating scan of right forearm, subject 000
    │       └── LH_rotation.h5 # US frames and associated transformations (from tracker tool space to optical camera space) in rotating scan of left forearm, subject 000
    │   
    │   ├── 001/
    │       ├── RH_rotation.h5 # US frames and associated transformations (from tracker tool space to optical camera space) in rotating scan of right forearm, subject 001
    │       └── LH_rotation.h5 # US frames and associated transformations (from tracker tool space to optical camera space) in rotating scan of left forearm, subject 001
    │   
    │   ├── ...
    │
    │
    ├── landmarks/
    │   ├── landmark_000.h5 # landmarks in scans of subject 000
    │   ├── landmark_001.h5 # landmarks in scans of subject 001
    │   ├── ...
    │
    └── calib_matrix.csv # calibration matrix

```

* Folder `frames_transfs`: contains 50 folders (one subject per folder), each with two scans. Each .h5 file corresponds to one scan, storing image and transformation of each frame within this scan. Key-value pairs and name of each .h5 file are explained below. 
    * `frames` - All frames in the scan; with a shape of [N,H,W], where N refers to the number of frames in the scan, H and W denote the height and width of a frame.

    * `tforms` - All transformations in the scan; with a shape of [N,4,4], where N is the number of frames in the scan, and the transformation matrix denotes the transformation from tracker tool space to camera space. 

    * Notations in the name of each .h5 file: `RH`: right arm; `LH`: left arm. For example, `RH_rotating.h5` denotes a rotating scan on the right forearm. 

* Folder `landmarks`: contains 50 .h5 files. Each corresponds to one subject, storing coordinates of landmarks for 2 scans of this subject. For each scan, the coordinates are stored in numpy array with a shape of [100,3]. The first column indicates the frame index (starting from 0), while the second and third columns represent the landmark coordinates in the image coordinate system (starting from 1, to maintain consistency with the calibration process).

* Calibration matrix: The calibration matrix was obtained using a pinhead-based method. The `scaling_from_pixel_to_mm` and `spatial_calibration_from_image_coordinate_system_to_tracking_tool_coordinate_system` are provided in the “calib_matrix.csv”, where `scaling_from_pixel_to_mm` is the scale between image coordinate system (in pixel) and image coordinate system (in mm), and `spatial_calibration_from_image_coordinate_system_to_tracking_tool_coordinate_system` is the rigid transformation between image coordinate system (in mm) to tracking tool coordinate system. Please refer to an example where this calibration matrix is read and used in the baseline code <a href="https://github.com/QiLi111/TUS-REC2025-Challenge_baseline/blob/df325edb0f3ae07f2f8eba993b4dee74fc608de1/train.py#L65" target="_blank">here</a>.

* Additional training and validation data (optional) come from previous challenge (TUS-REC2024), on the same cohort but with different scanning protocols. The patient IDs are consistent across datasets of TUS-REC2024 and TUS-REC2025 to ensure participants can properly account for data distribution when incorporating TUS-REC2024 data.
    * <a href="https://zenodo.org/doi/10.5281/zenodo.11178508" target="_blank">Training data (Part 1)</a>
    * <a href="https://zenodo.org/doi/10.5281/zenodo.11180794" target="_blank">Training data (Part 2)</a>
    * <a href="https://zenodo.org/doi/10.5281/zenodo.11355499" target="_blank">Training data (Part 3)</a>
    * <a href="https://zenodo.org/doi/10.5281/zenodo.12979481" target="_blank">Validation data</a>

## Training Code Usage Guide 

### Instruction
This repository provides an example framework for freehand US pose regression, including usage of various types of predictions and labels (see <a href="https://github.com/QiLi111/TUS-REC2025-Challenge_baseline/blob/main/utils/transform.py" target="_blank">transformation.py</a>). Please note that the network used here is small and simplified for demonstration purposes.

For instance, the network can predict the transformation between two US frames as 6 DOF "parameter". The loss could be calculated as the point distance between ground-truth-transformation-transformed points and predicted-transformation-transformed points, by transforming 4*4 "transform" and 6DOF "parameter" to "point" using functions <a href="https://github.com/QiLi111/TUS-REC2025-Challenge_baseline/blob/df325edb0f3ae07f2f8eba993b4dee74fc608de1/utils/transform.py#L82" target="_blank">to_points</a> and <a href="https://github.com/QiLi111/TUS-REC2025-Challenge_baseline/blob/df325edb0f3ae07f2f8eba993b4dee74fc608de1/utils/transform.py#L241" target="_blank">parameter_to_point</a>. The steps below illustrate an example of training a pose regression model and generating four displacement vectors. 
<!-- > [!NOTE]   -->
<!-- > * The ground truth transformation represents the mapping from one image coordinate system (in millimeters) to another, as exemplified by the function <a href="https://github.com/QiLi111/TUS-REC2025-Challenge_baseline/blob/df325edb0f3ae07f2f8eba993b4dee74fc608de1/utils/transform.py#L86" target="_blank">to_transform_t2t</a>. -->
<!-- > * The model trained with labels defined above is independent of the rigid part in calibration matrix, and only dependent of the scaling. That is to say, the trained model is independent of the relative position between the tracker tool and the probe, and only dependent of the configuration of the probe.  -->

<!-- For more information about the algorithms, refer to [Prevost et al. 2018](https://doi.org/10.1016/j.media.2018.06.003) and [Li et al. 2023](https://doi.org/10.1109/TBME.2023.3325551). -->

### Steps to run the code
#### 1. Clone the repository.
```
git clone https://github.com/QiLi111/TUS-REC2025-Challenge_baseline.git
```

#### 2. Navigate to the root directory.
```
cd TUS-REC2025-Challenge_baseline
```


#### 3. Install conda environment.

``` bash
conda create -n freehand-US python=3.9.13
conda activate freehand-US
pip install -r requirements.txt
conda install pytorch3d --no-deps -c pytorch3d
```
If you encounter a "Segmentation fault" error during the installation of pytorch3d, please refer to this <a href="https://github.com/facebookresearch/pytorch3d/issues/1891" target="_blank">link</a>.


<!-- #### 4. Create directory.
```
mkdir data
``` -->

#### 4. Download data <a href="https://zenodo.org/records/15224704" target="_blank">here</a> and put `Freehand_US_data_train_2025.zip` into root directory `TUS-REC2025-Challenge_baseline`.

#### 5. Unzip `Freehand_US_data_train_2025.zip` into `./data` directory.

```
unzip Freehand_US_data_train_2025.zip -d ./data
```

#### 6. Make sure the data folder structure is the same as [Train Data Structure](#train-data-structure) above.

<!-- <a href="https://github.com/QiLi111/TUS-REC2025-Challenge_baseline?tab=readme-ov-file#training-data-structure" target="_blank">Training Data Structure</a> above. -->

#### 7. Train a model. 
``` bash
python3 train.py
```
#### 8. Generate DDF.
``` bash
python3 generate_DDF.py
```
> [!NOTE]
> * The definition of the four DDFs are explained as follows:
>     * `GP`: Global displacement vectors for all pixels. DDF from the current frame to the first frame, in mm. The first frame is regarded as the reference frame. The DDF should be in numpy array format with a shape of [N-1,3,307200] where N-1 is the number of frames in that scan (excluding the first frame), "3" denotes “x”, “y”, and “z” axes, respectively, and 307200 is the number of all pixels in a frame. The order of the flattened 307200 pixels can be found in function <a href="https://github.com/QiLi111/TUS-REC2025-Challenge_baseline/blob/df325edb0f3ae07f2f8eba993b4dee74fc608de1/utils/plot_functions.py#L5" target="_blank">reference_image_points</a>.
>     * `GL`: Global displacement vectors for landmarks, in mm. The DDF should be in numpy array format with a shape of [3,100], where 100 is the number of landmarks in a scan.
>     * `LP`: Local displacement vectors for all pixels. DDF from current frame to the previous frame, in mm. The previous frame is regarded as the reference frame. The DDF should be in numpy array format with a shape of [N-1,3,307200], where N-1 is the number of frames in that scan (excluding the first frame), "3" denotes “x”, “y”, and “z” axes, respectively, and 307200 is the number of all pixels in a frame. The order of the flattened 307200 pixels can be found in function <a href="https://github.com/QiLi111/TUS-REC2025-Challenge_baseline/blob/df325edb0f3ae07f2f8eba993b4dee74fc608de1/utils/plot_functions.py#L5" target="_blank">reference_image_points</a>.
>     * `LL`: Local displacement vectors for landmarks, in mm. The DDF should be in numpy array format with a shape of [3,100], where 100 is the number of landmarks in a scan.
> * We have provided two functions, which can generate four DDFs from global and local transformations, in <a href="https://github.com/QiLi111/TUS-REC2025-Challenge_baseline/blob/main/utils/Transf2DDFs.py" target="_blank">Transf2DDFs.py</a>.
> * The order of the four DDFs and the order of 307200 pixels cannot be changed and they must all be numpy arrays. Please ensure your prediction does not have null values. Otherwise, the final score could not be generated.
> * If you want to see the plotted trajectories, please uncomment <a href="https://github.com/QiLi111/TUS-REC2025-Challenge_baseline/blob/01115ed7708f6300cf9edc0f4bdad02b13d5e5c3/generate_DDF.py#L51" target="_blank">this line</a>. 
> * Ensure sufficient RAM is available, as the DDF generation process is performed on the CPU. Insufficient memory may result in a "killed" message due to an out-of-memory error. You can use the `dmesg` command to check detailed system logs related to this issue. Alternatively, to reduce memory usage, consider generating DDFs in chunks. If your GPU has at least 30 GB of memory, you may also generate DDFs on the GPU by moving the inputs to the GPU within the two functions: <a href="https://github.com/QiLi111/TUS-REC2025-Challenge_baseline/blob/df325edb0f3ae07f2f8eba993b4dee74fc608de1/utils/Transf2DDFs.py#L5" target="_blank">cal_global_ddfs</a> and <a href="https://github.com/QiLi111/TUS-REC2025-Challenge_baseline/blob/df325edb0f3ae07f2f8eba993b4dee74fc608de1/utils/Transf2DDFs.py#L36C5-L36C19" target="_blank">cal_local_ddfs</a>. However, please ensure that the total GPU memory usage remains below 32 GB during testing.

## Data Usage Policy
The training and validation data provided may be utilized within the research scope of this challenge and in subsequent research-related publications. However, commercial use of the training and validation data is prohibited. In cases where the intended use is ambiguous, participants accessing the data are requested to abstain from further distribution or use outside the scope of this challenge. Please refer to <a href="https://github-pages.ucl.ac.uk/tus-rec-challenge/policies.html" target="_blank">Challenge Rules & Policies</a> for detailed data usage policy.

We are planning to submit a challenge paper including the analysis of the dataset and the results. Members of the top participating teams will be invited as co-authors. The invited teams will be announced after the challenge event and would depend on the number of participating teams. The challenge organizers determine the order of the authors in the joint challenge paper. The participating teams can publish their results separately but only after a publication of the joint challenge paper (expected by end of 2026). If you have any queries about the publication policy, please contact us. Once the challenge paper from the organizing team is published, the participants should cite this challenge paper.

After we publish the summary paper of the challenge, if you use our dataset in your publication, please cite the summary paper (reference will be provided once published) and some of the follwing articles: 
* Qi Li, Ziyi Shen, Qianye Yang, Dean C. Barratt, Matthew J. Clarkson, Tom Vercauteren, and Yipeng Hu. "Nonrigid Reconstruction of Freehand Ultrasound without a Tracker." In International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 689-699. Cham: Springer Nature Switzerland, 2024. doi: <a href="https://doi.org/10.1007/978-3-031-72083-3_64" target="_blank">10.1007/978-3-031-72083-3_64</a>
* Qi Li, Ziyi Shen, Qian Li, Dean C. Barratt, Thomas Dowrick, Matthew J. Clarkson, Tom Vercauteren, and Yipeng Hu. "Long-term Dependency for 3D Reconstruction of Freehand Ultrasound Without External Tracker." IEEE Transactions on Biomedical Engineering, vol. 71, no. 3, pp. 1033-1042, 2024. doi: <a href="https://doi.org/10.1109/TBME.2023.3325551" target="_blank">10.1109/TBME.2023.3325551</a>
* Qi Li, Ziyi Shen, Qian Li, Dean C. Barratt, Thomas Dowrick, Matthew J. Clarkson, Tom Vercauteren, and Yipeng Hu. "Trackerless freehand ultrasound with sequence modelling and auxiliary transformation over past and future frames." In 2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI), pp. 1-5. IEEE, 2023. doi: <a href="https://doi.org/10.1109/ISBI53787.2023.10230773" target="_blank">10.1109/ISBI53787.2023.10230773</a>
<!-- * Qi Li, Ziyi Shen, Qian Li, Dean C. Barratt, Thomas Dowrick, Matthew J. Clarkson, Tom Vercauteren, and Yipeng Hu. "Privileged Anatomical and Protocol Discrimination in Trackerless 3D Ultrasound Reconstruction." In International Workshop on Advances in Simplifying Medical Ultrasound, pp. 142-151. Cham: Springer Nature Switzerland, 2023. doi: <a href="https://doi.org/10.1007/978-3-031-44521-7_14" target="_blank">https://doi.org/10.1007/978-3-031-44521-7_14</a> -->


## Organizers

[Qi Li](mailto:qi.li.21@ucl.ac.uk), University College London

Yuliang Huang, University College London

Shaheer U. Saeed, University College London

Dean C. Barratt, University College London

Matthew J. Clarkson, University College London

Tom Vercauteren, King's College London

Yipeng Hu, University College London

<!-- Challenge Contact E-Mail: [`qi.li.21@ucl.ac.uk`](mailto:qi.li.21@ucl.ac.uk) -->

## Sponsors

<div >
  <a href="http://ucl.ac.uk/hawkes-institute/" target="_blank"><img style="padding: 30px;" src="img/UCL-Hawkes-Institute-BLACK.jpg" width=140px></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   
  <a href="https://conferences.miccai.org/2025/en/" target="_blank"><img style="padding: 30px;" src="img/miccai2025-logo.png" width=130px></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   
  <a href="https://miccai-ultrasound.github.io/#/asmus25" target="_blank"><img style="padding: 30px;" src="img/asmus.png" width=115px></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  
  <a href="https://miccai.org/index.php/special-interest-groups/sig/" target="_blank"><img style="padding: 30px;" src="img/SIGMUS.png" width=220px></a>   
</div>