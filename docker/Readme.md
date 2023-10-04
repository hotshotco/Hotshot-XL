# Setup

This docker file is for the **environment only**. This is to keep the docker image as small as possible!

## Quickstart

Hotshot have their own docker image you can use directly:
```
docker pull hotshotapp/hotshot-xl-env:latest
```

Or you can build it yourself

```
cd docker
docker build -t hotshotapp/hotshot-xl-env:latest .
```

## Running the docker image

We recommend storing the weights locally on your machine. That way the weights persist if you kill the container!

- Install the models to a folder locally (Optional)
     ```
    cd /path/to/models
    git lfs install
    git clone https://huggingface.co/hotshotco/Hotshot-XL
     ```
- Run the docker from the project root
    - **Linux**
    ```
    docker run -it --gpus=all --rm -v $(pwd):/local -v /path/to/models:/models hotshotapp/hotshot-xl-env:latest
    ```
    - **Windows (Powershell)**
    ```
    docker run -it --gpus=all --rm -v ${PWD}:/local -v C:\path\to\models:/models hotshotapp/hotshot-xl-env:latest
    ```
  
If you want to download the models from within the container itself then you do not need to map the volumes and ` -v /path/to/models:/models` can be removed.

**Note**: Ensure you have NVIDIA Docker runtime installed if you want to utilize GPU support with `--gpus=all`.
