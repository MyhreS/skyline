# Skyline

This is the codebase for my master thesis, following a naming scheme inspired by cloud formations and the sky.

## Setup

This project uses Python 3.10.11, a requirement due to server constraints for training the models. To ensure consistent results, it's recommended to use this specific version.

### Setting Up Python Version

First, install pyenv if you don't have it already:

```
brew install pyenv
```

Next, install and set the specific Python version:

```
pyenv install 3.10.11
pyenv global 3.10.11
```

Run the following to check the version:

```
python3.10 --version
```

If the command doesn't work, add pyenv to your path:

```
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
```

Apply the changes (this example is for zsh):

```
source ~/.zshrc
```

### Setting Up the Virtual Environment

This project uses `venv`. Set it up as follows:

```
python3.10 -m venv skylinevenv
```

Activate the environment:

```
source skylinevenv/bin/activate
```

Install the dependencies:

```
pip install -r requirements.txt
```

### Specify the Path to the Data

To train new models, specify the path to your data. Create a `.env` file in the root directory, including the name and path to the dataset folders.

### Running a File

With the virtual environment activated, run any Python file:

```
python {filename}
```

## Theme: Cloud Formations

Consider these names inspired by cloud formations:

1. **Cirrus**
2. **Cumulus**

## Setup Docker Tensorflow on Server

Link to guide: https://www.youtube.com/watch?v=jdip_6vTw0s&t=137s

### Step 1: Install docker

```
sudo apt update
sudo apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker
```

### Step 2: Check that you have installed docker correctly

```
docker --version
```

### Step 3: Install nvidia toolkit container

Link to webside: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

```
sudo apt-get update
```

```
sudo apt-get install -y nvidia-container-toolkit
```

### Step 4: Configure docker to use the nvidia-container:

```
sudo nvidia-ctk runtime configure --runtime=docker
```

```
sudo systemctl restart docker
```

### Step 5: Install tensorflow docker image

Link to website: https://hub.docker.com/r/tensorflow/tensorflow

```
docker pull tensorflow/tensorflow:nightly-gpu
```
