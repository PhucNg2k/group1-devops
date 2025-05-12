#!/bin/bash
sudo apt install apt-transport-https ca-certificates curl software-properties-common -y
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
echo "--------------------- DOCKER VERSION----------------------------"
sudo docker --version
sudo systemctl enable docker
echo "--------------------- DOCKER STATUS----------------------------"
sudo systemctl status docker

# Install Docker Compose
echo "--------------------- INSTALLING DOCKER COMPOSE ----------------------------"
# Docker Compose is now included in docker-compose-plugin package installed above,
# but we'll also add the standalone version for compatibility
COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep 'tag_name' | cut -d\" -f4)
sudo curl -L "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Display Docker Compose version
echo "--------------------- DOCKER COMPOSE VERSION----------------------------"
docker-compose --version
