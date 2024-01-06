# UP Sticker

Detect UP Stickers from cars using image processing techniques

Ciron | Corrales | Dayag

## First time setup

Instructions are specific for **PYTHON 3.11**

create a virtual environment

```sh
python -m venv env
```

activate the virtual environment

```sh
#in Linux
source env/bin/activate

#in Windows must use Command Prompt
.\env\Scripts\activate.bat
# if successful the path should look something like this
# (env) C:\...\...>
```

## Install the dependencies

### SIMPLE

```sh
pip install ultralytics
```

https://download.pytorch.org/whl/torch/

https://download.pytorch.org/whl/torchvision/

### CPU ONLY LINUX
```sh
pip install https://download.pytorch.org/whl/cpu/torch-2.1.2%2Bcpu-cp311-cp311-linux_x86_64.whl
pip install https://download.pytorch.org/whl/cpu/torchvision-0.16.2%2Bcpu-cp311-cp311-linux_x86_64.whl
pip install -r cpu.txt
```

### CPU ONLY WINDOWS
```sh
pip install https://download.pytorch.org/whl/cpu/torch-2.1.2%2Bcpu-cp311-cp311-win_amd64.whl
pip install https://download.pytorch.org/whl/cpu/torchvision-0.16.2%2Bcpu-cp311-cp311-win_amd64.whl
pip install -r cpu.txt
```
