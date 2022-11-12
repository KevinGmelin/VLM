# Vision and Language Manipulation

This project is being developed by Kevin Gmelin, Sabrina Shen, Jaekyung Song, and Kwangkyun Kim (Bruce) as part of the final project for Carnegie Mellon University's 11785 Intro to Deep Learning class.

The goal is to develop an agent that can learn to combine vision and language to conduct manipulation tasks. The benchmark for collecting training data and evaluating our agents is [VLMbench](https://github.com/eric-ai-lab/vlmbench).

# AWS Setup
To install on AWS, first run AWS_SETUP.sh:

    sh AWS_SETUP.sh

Then reboot:

    sudo reboot

To render copellia sim using virtual gl, make sure to start an X server and set the DISPLAY environment variable:

    nohup sudo X &
    export DISPLAY=:0.0

To download the VLMbench pick training dataset, use gdown:

    gdown https://drive.google.com/uc?id=1TmSoJ3BufCITsUturTKHft8fXxia8SFk
