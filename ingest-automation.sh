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
/opt/anaconda3/envs/chatbot/bin/python ingest.py

# Change user permissions on DB
chmod -R 777 /chatbot/source/api/DB

# Start the Gunicorn service
systemctl start gunicorn.service
