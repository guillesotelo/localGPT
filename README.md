# Local LLM API

## Server configuration

### Apache routing

This is how /etc/apache2/sites-enabled/chatbot.conf (or 000-default.conf) should look like:

```bash
<VirtualHost 10.55.101.133:80>
    ServerName hpchatbot.server-name.net
    ServerAlias hpchatbot.server-name.net
    Redirect permanent / https://hpchatbot.server-name.net/
</VirtualHost>

<IfModule mod_ssl.c>
    <VirtualHost 10.55.101.133:443>
        ServerName hpchatbot.server-name.net
        ServerAdmin guillermo.sotelo@server-name.com

        # Enable Proxying
        ProxyPreserveHost On
        ProxyRequests Off

        # Proxy API requests to Node.js API (Port 5000)
        ProxyPass /api http://localhost:5000/api
        ProxyPassReverse /api http://localhost:5000/api

        # Proxy all other requests to React app (Port 3000)
        ProxyPass / http://localhost:3000/
        ProxyPassReverse / http://localhost:3000/

        # Error and Access Logs
        ErrorLog ${APACHE_LOG_DIR}/hpchatbot-error.log
        CustomLog ${APACHE_LOG_DIR}/hpchatbot-access.log combined

        # SSL Configuration
        SSLEngine on
        SSLCertificateFile /var/lib/hpchatbot/ssl/CAbundle2024.cer
        SSLCertificateKeyFile /var/lib/hpchatbot/ssl/PrivateKey.key
        SSLCertificateChainFile /var/lib/hpchatbot/ssl/CAbundle2024.cer
        SSLCACertificatePath /etc/ssl/certs/

        # Optional: Strict HTTPS Enforcement
        <FilesMatch "\.(cgi|shtml|phtml|php)$">
            SSLOptions +StdEnvVars
        </FilesMatch>
        <Directory /usr/lib/cgi-bin>
            SSLOptions +StdEnvVars
        </Directory>
    </VirtualHost>
</IfModule>
```

Then making sure http, proxy_http and ssl are enabled:

```bash
sudo a2enmod proxy proxy_http ssl
sudo systemctl restart apache2
```

### Gunicorn

We use Gunicorn for serving the API in a production-ready environment:

- Create a systemd service file, e.g., /etc/systemd/system/gunicorn.service

```bash
[Unit]
Description=Gunicorn instance to serve your API
After=network.target

[Service]
User=your_user
Group=your_group
WorkingDirectory=/path/to/your/app
ExecStart=/home/guillermo/anaconda3/envs/myenv/bin/gunicorn --bind 0.0.0.0:5000 run_api:app --workers 1 --threads 1 --timeout 240
Restart=always

[Install]
WantedBy=multi-user.target
```

Make sure to update WorkingDirectory and ExactStart accordingly.
Then:

```bash
sudo systemctl daemon-reload
```

```bash
sudo systemctl start gunicorn
```

```bash
sudo systemctl enable gunicorn
```

```bash
sudo systemctl status gunicorn
```

View live logs using:

```bash
journalctl -u gunicorn.service -f
```

## Instalation & Testing with CPU

Default values from constants.py are for GPU. Create an .env file with the values you want for CPU for example:

```bash
   CONTEXT_WINDOW_SIZE=2048
   MAX_NEW_TOKENS=512
   N_GPU_LAYERS=0
   N_BATCH=128
   EMBEDDING_MODEL_NAME=hkunlp/instructor-large
```

A successfull test was made with the following stack:

- Model: TheBloke--Mistral-7B-Instruct-v0.2-GGUF (mistral-7b-instruct-v0.2.Q4_K_M.gguf)
- Embeddings: "hkunlp/instructor-large"
- huggingface_hub==0.23.2

If problems arise with llamaCpp run:

```bash
   pip install --force-reinstall --no-cache-dir llama-cpp-python==0.2.58
```

- Update model_path in load_model
- Playing with adding parameter `n_ctx=2048``and other numbers on LlamaCpp()
- Playing around with system_prompt starting value

## Run

To run the API start with:

```bash
   conda activate localGPT
   python run_localGPT_API.py
```

For SSL related issues run this command and try running the API again:

```bash
   export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
```

### Fix broken dependencies

Run:

```bash
   pip install -r requirements.txt --force-reinstall
```

If still dependency errors arise:

```bash
   pip uninstall pyyaml markdown-it-py langchain langsmith
```

and:

```bash
   pip install pyyaml==5.1
   pip install markdown-it-py==2.1.0  # Compatible with mdit-py-plugins and myst-parser
   pip install langchain==0.2.16
   pip install langsmith==0.1.112 # Or the ones to be fixed`
```

Finally check: `pip check` to confirm dependencies ar ok.

When all is working, freeze into requirements:

```bash
   pip freeze > requirements.txt
```

## Installation on GPU

### CUDA Installation

This was tested using UBUNTU 22.04 on a VM from Azure with a Tesla T4 GPU.

Disable Secure Boot and install CUDA drivers (if it's a VM, login on vm provider, stop instance, disable Secure Boot from the configuration panel and start the instance again).

Check Secure Boot state:

```bash
   mokutil --sb-state
```

Run these commands separately:

```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
   sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
   wget https://developer.download.nvidia.com/compute/cuda/12.5.0/local_installers/cuda-repo-ubuntu2204-12-5-local_12.5.0-555.42.02-1_amd64.deb
   sudo dpkg -i cuda-repo-ubuntu2204-12-5-local_12.5.0-555.42.02-1_amd64.deb
   sudo cp /var/cuda-repo-ubuntu2204-12-5-local/cuda-*-keyring.gpg /usr/share/keyrings/
   sudo apt-get update
   sudo apt-get -y install cuda-toolkit-12-5
```

```bash
   sudo reboot
```

And now verify installation of nvidia and cuda drivers:

```bash
   nvidia-smi
```

Install cuda-drivers (not sure if needed)

```bash
   sudo apt-get install -y cuda-drivers
```

Add CUDA paths:

```bash
   sudo nano /home/local/VCCNET/<username>/.bashrc # Or /home/<username>/.bashrc
```

Add the following paths at the end (match the CUDA version used):

```bash
export PATH="/usr/local/cuda-12.5/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.5/lib64:$LD_LIBRARY_PATH"
```

Then opening new bash window or after running `bash`:

```bash
   nvcc --version # Should return Nvidia compiler information
```

#### Usefull commands

List cuda packages:

```bash
   pip list | grep cuda
```

List cuda versions:

```bash
   sudo update-alternatives --display cuda
```

Select cuda version:

```bash
   sudo update-alternatives --config cuda
```

### API Installation

- Download and install Conda

```bash
   wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
```

```bash
   sudo chmod 777 Anaconda3-2024.10-1-Linux-x86_64.sh
   sudo chmod -x Anaconda3-2024.10-1-Linux-x86_64.sh
```

```bash
   ./Anaconda3-2024.10-1-Linux-x86_64.sh
```

Test typing `conda`. You might need to verify you are in bash with `bash` command first.

- Create python environment using verion 3.10

```bash
conda create -n localGPT python=3.10.0
conda activate localGPT
```

```shell
pip install -r requirements.txt
```

- Install LlamaCPP

For `NVIDIA` GPUs support, use `cuBLAS`

```shell
   CMAKE_ARGS="-DGGML_CUDA=on -DCUDA_PATH=/usr/local/cuda-12.5 -DCUDAToolkit_ROOT=/usr/local/cuda-12.5 -DCUDAToolkit_INCLUDE_DIR=/usr/local/cuda-12/include -DCUDAToolkit_LIBRARY_DIR=/usr/local/cuda-12.5/lib64" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir
```

Option 2 (this one worked with cuda 12.5):

```shell
   CMAKE_ARGS="-DGGML_CUDA=on -DLLAVA_BUILD=off" pip install -U llama-cpp-python --force-reinstall --no-cache-dir
```
