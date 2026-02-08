# Documentation setup de la jetson

Ce projet a pour but d'utiliser la jetson nano 2GB developper kit afin de faire de la reconnaissance de champion league of legends en temps réel. 
Pour cela nous utiliserons un modèle YOLO. 

Sommaire : 
- Configuration la jetson
- Faire tourner un modèle deja entrainé sur le dataset COCO 
- Entrainer son propre modèle
- Mettre en place une connection RTSP entre la jetson et l'ordinateur faisant tourner lol

## Première partie : Configuration de la jetson
- Flasher un OS sur la carte SD 
- Création d'un environnement python
- Installation d'OpenCV
- Installation de pytorch et de torchvision

### Flasher cette OS sur une carte SD : https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-2gb-devkit#write

### Création d'un environnement avec python

- Installer pip le package manager de python

```
sudo apt-get install python3-pip
```

- Installer virtualenv qui nous permettra de créer des environnements virtuels
```
pip3 install virtualenv
```

- Créer un environnement virtuel 
```
python3 -m virtualenv -p python3 env --system-site-packages   
```
| Segment | Rôle |
| :--- | :--- |
| `python3 -m virtualenv` | Lance le module `virtualenv` en utilisant l'interpréteur Python 3. |
| `-p python3` | Force l'utilisation d'une version spécifique de Python (ici, Python 3.6.9) pour l'environnement. |
| `env` | C'est le nom du dossier qui sera créé. Vous y trouverez l'exécutable Python et vos futures bibliothèques. |
| `--system-site-packages` | Le point clé : Autorise l'environnement virtuel à accéder aux packages installés au niveau du système (ceux installés via `apt` ou `pip` global). Nous faisons cela pour que notre environnement virtuel est accès à opencv qui est nativement installé sur l'OS de la jetson|

- Activer l'environnement 
```
source env/bin/activate 
```

- Désactiver l'environnement 

```
deactivate
```

### Création d'un swap file

Notre jetson ne possédant que 2GB de RAM, ceci est très limitant surtout lorsque nous voudrons charger des modèles qui font plus de 2GB. Il est donc vivement conseillé de créer un swap file afin de prévenir des problèmes du type out of memory.

```
sudo fallocate -l 4G /var/swapfile 
```


```
sudo chmod 600 /var/swapfile
```

```
sudo mkswap /var/swapfile
```

```
sudo swapon /var/swapfile
```

```
sudo bash -c 'echo "/var/swapfile swap swap defaults 0 0"  >> /etc/fstab’ 
```

Après cela il faudra reboot la jetson 
```
sudo reboot
```

Et vérifier si le swapFile à correctement été créer 

```
free -h
```

### Installation d'openCV 4.5 avec support cuda 
openCV est une référence dans le traitement d'image, c'est une dépendances de yolo. Cette installation est manuel et prend pas mal de temps. 

Tout d'abord il faut installer toutes les dépendances de openCV : 
```
sudo sh -c "echo '/usr/local/cuda/lib64' >> /etc/ld.so.conf.d/nvidia-tegra.conf" && \
sudo ldconfig && \
sudo apt-get update && \
sudo apt-get install -y build-essential cmake git unzip pkg-config \
libjpeg-dev libpng-dev libtiff-dev \
libavcodec-dev libavformat-dev libswscale-dev \
libgtk2.0-dev libcanberra-gtk* \
python3-dev python3-numpy python3-pip \
libxvidcore-dev libx264-dev libgtk-3-dev \
libtbb2 libtbb-dev libdc1394-22-dev \
libv4l-dev v4l-utils \
libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
libavresample-dev libvorbis-dev libxine2-dev \
libfaac-dev libmp3lame-dev libtheora-dev \
libopencore-amrnb-dev libopencore-amrwb-dev \
libopenblas-dev libatlas-base-dev libblas-dev \
liblapack-dev libeigen3-dev gfortran \
libhdf5-dev protobuf-compiler \
libprotobuf-dev libgoogle-glog-dev libgflags-dev
```

Télécharger openCV

```
cd ~ && wget -O opencv.zip https://github.com/opencv/opencv/archive/4.5.1.zip && wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.5.1.zip && unzip opencv.zip && unzip opencv_contrib.zip && mv opencv-4.5.1 opencv && mv opencv_contrib-4.5.1 opencv_contrib && rm opencv.zip opencv_contrib.zip && cd ~/opencv && mkdir -p build && cd build && cmake -D CMAKE_BUILD_TYPE=RELEASE       -D CMAKE_INSTALL_PREFIX=/usr       -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules       -D EIGEN_INCLUDE_PATH=/usr/include/eigen3       -D WITH_OPENCL=OFF       -D WITH_CUDA=ON       -D CUDA_ARCH_BIN=5.3       -D CUDA_ARCH_PTX=""       -D WITH_CUDNN=ON       -D WITH_CUBLAS=ON       -D ENABLE_FAST_MATH=ON       -D CUDA_FAST_MATH=ON       -D OPENCV_DNN_CUDA=ON       -D ENABLE_NEON=ON       -D WITH_QT=OFF       -D WITH_OPENMP=ON       -D WITH_OPENGL=ON       -D BUILD_TIFF=ON       -D WITH_FFMPEG=ON       -D WITH_GSTREAMER=ON       -D WITH_TBB=ON       -D BUILD_TBB=ON       -D BUILD_TESTS=OFF       -D WITH_EIGEN=ON       -D WITH_V4L=ON       -D WITH_LIBV4L=ON       -D OPENCV_ENABLE_NONFREE=ON       -D INSTALL_C_EXAMPLES=OFF       -D INSTALL_PYTHON_EXAMPLES=OFF       -D BUILD_NEW_PYTHON_SUPPORT=ON       -D BUILD_opencv_python3=TRUE       -D OPENCV_GENERATE_PKGCONFIG=ON       -D BUILD_EXAMPLES=OFF .. && make -j4 && sudo rm -rf /usr/include/opencv4/opencv2 && sudo make install && sudo ldconfig && make clean && sudo apt-get update
```

Pour véririfer l'installation 

```
python3 -c "import cv2; print('Version:', cv2.__version__);"
```

### Installation de PyTorch v1.8 - torchvision v0.9.0

- Installation de pytorch 
```
wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev libomp-dev
pip3 install 'Cython<3'
pip3 install numpy torch-1.8.0-cp36-cp36m-linux_aarch64.whl
```

- Vérification de la version de pytorch

```
python3 -c "import torch; print('Version:', torch.__version__);"
```

- Installation de torchvision 

```
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.9.0 https://github.com/pytorch/vision torchvision
cd torchvision
pip3 install --user "pillow<8.5.0"
export BUILD_VERSION=0.9.0 
python3 setup.py install --user
```

- Vérification de la version de torchvision

```
python3 -c "import torchvision; print(torchvision.__version__)"
```

