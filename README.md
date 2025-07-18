# Local LLM API

## SSH Connection

If you are going to work on the company's server, then first [connect through SSH](#ssh).

## Instalation for CPU

### Conda Installation

We are going to use Conda for our Python environment. Check if you have it by typing `conda`. You might need to verify you are in bash with `bash` command first.

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
conda create -n veronica python=3.10.0
conda activate veronica
```

### Requirements

```shell
pip install -r requirements.txt
```

If problems arise with llamaCpp run:

```bash
   pip install --force-reinstall --no-cache-dir llama-cpp-python==0.2.58
```

You might run over a bunch of dependency errors. If that happens refer to [Fix Broken Dependencies](#Fix-broken-dependencies).
Some manual work could be necessary to fix some of them. Take a look at possible error logs and fix them accordingly based on your local.

### Configure and Run the API

Default values from constants.py are for GPU. Create an .env file with the values you want for CPU which will overwrite defaults, for example you can start with:

```bash
   RETRIEVE_K_DOCS=5
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   CONTEXT_WINDOW_SIZE=1024
   MAX_NEW_TOKENS=256
   N_GPU_LAYERS=0
   N_BATCH=128
   EMBEDDING_MODEL_NAME=hkunlp/instructor-large
```

A successfull test was made with the following stack:

- Model: TheBloke--Mistral-7B-Instruct-v0.2-GGUF (mistral-7b-instruct-v0.2.Q4_K_M.gguf)
- Embeddings: "hkunlp/instructor-large"
- huggingface_hub==0.23.2

To run the api [follow this](#run-api).

## Installation for GPU

### CUDA Installation

This was tested using UBUNTU 22.04 on a VM from Azure with a Tesla T4 GPU. If you are using different hardware, then you might want to update some variables in `constants.py`.

Disable Secure Boot and install CUDA drivers (if it's a VM, login on vm provider, stop VM instance, disable Secure Boot from the Azure configuration panel and start the instance again).

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

Install cuda-drivers (not sure if needed):

```bash
   sudo apt-get install -y cuda-drivers
```

Add CUDA paths:

```bash
   sudo nano /home/local/VCCNET/<username>/.bashrc # Or /home/<username>/.bashrc
```

Add the following paths at the end (match the used CUDA version, 12.5 in this example):

```bash
export PATH="/usr/local/cuda-12.5/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.5/lib64:$LD_LIBRARY_PATH"
```

Then opening new bash window or after running `bash`:

```bash
   nvcc --version # Should return Nvidia compiler information
```

#### Usefull Commands

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

### Requirements Installation

#### Conda

Install Conda and setup Python environment first, [following these instructions](#conda-installation).

#### PIP

```shell
pip install -r requirements.txt
```

- Install LlamaCPP

For `NVIDIA` GPUs support, use `cuBLAS`

```shell
   CMAKE_ARGS="-DGGML_CUDA=on -DLLAVA_BUILD=off" pip install -U llama-cpp-python --force-reinstall --no-cache-dir
```

#### Fix Broken Dependencies

Run:

```bash
   pip install -r requirements.txt --force-reinstall
```

If still dependency errors arise, try:

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

After all is working, freeze requirements so we make sure we save our operative state:

```bash
   pip freeze > requirements.txt
```

## Run API

To run the API, make sure you are on your python environment first, then run the main script like this:

```bash
   conda activate veronica
   python run_api.py
```

You should see a bunch of logs from the model parameters and other debugging lines. At the end it will show "LLM loaded", meaning the API is ready and you can start using it from the [front end](#frontend).

### SSL Issues

For SSL related issues run this command and try running the API again:

```bash
   export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
```

If that doesn't work, try with empty path instead:

```bash
   export REQUESTS_CA_BUNDLE=
```

If the error is about MaxRetryError for a certain host like `cdn-lfs-us-1.hf.co`, it's a good practice to force HuggingFace mirror to be US based:

```python
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

## Frontend

### Run Node app

To run the React App for testing or development, stand on the UI folder and do the following:

```bash
npm run dev
```

This will compile the app and serve it at, if port not in use, http://localhost:3000, otherwise it will state it is in use and you can choose to serve the app over another one.
Now, with the server running in one side with CPU or GPU configs, you should be able to start using the chat.

#### Production

We use pm2 to control the app in production.
To run the application for the first time, go to `/chatbot/source/ui` and run this command:

```bash
pm2 delete all && pm2 start server.js
```

To check the logs: 

```bash
pm2 logs
```

This server serves the build folder wich is generated from a react app. We can deploy this by running this on the same folder:

```bash
npm run build
```

This will display a message on the frontend about a build taking place, and reload the in around 20 seconds (build ussually takes 15s).


### Apache (Veronica's Specific)

The chatbot frontend is served via Node and Apache service.

This is the working hpchatbot.conf in folder `/etc/apache2/sites-enabled` (make sure to update the IP with the server's specific):

```bash
<VirtualHost 10.55.101.133:80>
    ServerName server.url.com
    ServerAlias server.url.com
    Redirect permanent / https://server.url.com/
</VirtualHost>

<IfModule mod_ssl.c>
    <VirtualHost 10.55.101.133:443>
        ServerName server.url.com
        ServerAdmin mail@serverurl.com

        # Enable Proxying
        ProxyPreserveHost On
        ProxyRequests Off

        # Proxy API requests to API
        ProxyPass /api http://localhost:5000/api
        ProxyPassReverse /api http://localhost:5000/api

        # Proxy all other requests to React app
        ProxyPass / http://localhost:3000/
        ProxyPassReverse / http://localhost:3000/

        ErrorLog ${APACHE_LOG_DIR}/hpchatbot-error.log
        CustomLog ${APACHE_LOG_DIR}/hpchatbot-access.log combined

        # SSL Config
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
            AddHandler cgi-script .py
        </Directory>

        FileETag None
        Header unset ETag
        Header set Cache-Control "max-age=0, no-cache, no-store, must-revalidate"
        Header set Pragma "no-cache"
        Header set Expires "Wed, 21 Oct 2015 01:00:00 GMT"

    </VirtualHost>
</IfModule>

# for CGI-bin
<VirtualHost 10.55.101.133:8080>
    ServerAdmin mail@serverurl.com
    DocumentRoot /var/www/html
</VirtualHost>
```

Note: If the server was rebooted and apache2 service present errors, run:

```bash
sudo a2enmod headers
sudo systemctl restart apache2
```

Then making sure http, proxy_http and ssl are enabled:

```bash
sudo a2enmod proxy proxy_http ssl
sudo systemctl restart apache2
```

#### Gunicorn

We use Gunicorn for serving the API in a production-ready environment:

- Create a systemd service file, e.g., /etc/systemd/system/gunicorn.service

```bash
[Unit]
Description=Gunicorn instance to serve your API
After=network.target

[Service]
User=gsotelo
Group=domain^users
WorkingDirectory=/chatbot/source/api
ExecStart=/home/local/VCCNET/gsotelo/anaconda3/envs/localGPT/bin/gunicorn --bind 0.0.0.0:5000 run_api:app --workers 1 --threads 1 --timeout 300
Restart=always
StandardOutput=journal
StandardError=journal

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

#### Ingest automation

- Create a Shell Script: Create a script that stops the service, runs the process, and restarts the service.

```bash
#!/bin/bash

# Stop the Gunicorn service
systemctl stop gunicorn.service

# Navigate to the working directory
cd /chatbot/source/api || exit

# Delete the 'DB' folder if it exists
if [ -d "DB" ]; then
    rm -rf DB
    echo "DB folder deleted."
else
    echo "DB folder does not exist. Skipping deletion."
fi

# Run the ingestion script
/home/local/VCCNET/gsotelo/anaconda3/envs/localGPT/bin/python ingest.py

# Start the Gunicorn service
systemctl start gunicorn.service
```

- Save this script as /chatbot/source/api/ingest-automation.sh and make it executable:

```bash
chmod +x /chatbot/source/api/ingest-automation.sh
```

- Create a systemd Timer: Create two files: one for the service to run the script and another for the timer.

```bash
# Service file: /etc/systemd/system/ingest-automation.service
[Unit]
Description=Run ingest-automation tasks for Gunicorn

[Service]
Type=oneshot
ExecStart=/chatbot/source/api/ingest-automation.sh
```

```bash
# Timer file: /etc/systemd/system/ingest-automation.timer
[Unit]
Description=Run ingest-automation tasks daily at 6 AM

[Timer]
OnCalendar=*-*-* 06:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

- Enable and Start the Timer: Reload systemd and enable the timer.

```bash
sudo systemctl daemon-reload
sudo systemctl enable ingest-automation.timer
sudo systemctl start ingest-automation.timer
```

- Verify the Timer: Check the timer status to confirm it’s active:
  
```bash
sudo systemctl list-timers --all
```

#### Depoloyments

Right now we use SFTP protocols for updating code base and everything else on the server.
A recommended tool is FileZilla for Unix based systems (iOS / Ubuntu) or PuTTY for Windows.

We start a connection with the server's IP, username and password (ussually company's CDSID and password).

##### SSH

Connection to the server can be done with:

```bash
ssh cdsid@10.55.101.133 # Replace the IP with the actual server's IP if needed
```

If it's the first time connecting to this server, you will be prompted to trust the fingerprint. Respond yes and you will be connected and ready.

For easy use and for some Conda requirements, it is needed to use the `bash` CLI. So make sure by typing `bash` before you start working with Conda.