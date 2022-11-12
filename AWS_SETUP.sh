# Clone the vlm bench and VLM repos
cd ~
git clone https://github.com/eric-ai-lab/VLMbench.git
git clone https://github.com/KevinGmelin/VLM.git

# Get coppellia sim
wget https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
tar -xvf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
rm CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

# Get and install PYREP
git clone https://github.com/stepjam/PyRep.git
echo 'export COPPELIASIM_ROOT=~/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT' >> ~/.bashrc
echo 'export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT' >> ~/.bashrc
conda init bash
source ~/.bashrc
cd PyRep/
conda activate pytorch
pip3 install -r requirements.txt
pip3 install .
cd ..

# Setup Headless
sudo apt update
sudo apt-get install xorg libxcb-randr0-dev libxrender-dev libxkbcommon-dev libxkbcommon-x11-0 libavcodec-dev libavformat-dev libswscale-dev
sudo nvidia-xconfig -a --virtual=1280x1024
wget https://sourceforge.net/projects/virtualgl/files/2.5.2/virtualgl_2.5.2_amd64.deb/download -O virtualgl_2.5.2_amd64.deb --ca-directory=/etc/ssl/certs/
sudo dpkg -i virtualgl*.deb
rm virtualgl*.deb

# Install VLMbench
cd VLMbench/
conda activate pytorch
sed -i -e 's/opencv-python==4.2.0.32/opencv-python/g' requirements.txt
pip install -r requirements.txt
pip install absl-py kornia num2words
pip install -e .
cp ./simAddOnScript_PyRep.lua ../CoppeliaSim_Edu_V4_1_0_Ubuntu20_04/
cd ..

# Install gdown for downloading the VLMbench datasets
pip install gdown