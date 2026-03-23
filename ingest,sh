sudo systemctl stop gunicorn && sudo rm -rf DB

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate chatbot && python3 ingest.py

sudo systemctl start gunicorn && journalctl -u gunicorn -f