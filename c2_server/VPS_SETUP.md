# Google VPS Setup Guide (C2 Dashboard & SSL Tunnel)

Run these commands on your **Google Cloud VPS** to set up the dashboard, SSL, and the Chisel WebSocket proxy for Reverse SSH.

## 1. System Dependencies & Certbot
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv nginx certbot python3-certbot-nginx curl git
```

## 2. Get the SSL Certificate
*Make sure your domain `lalithadithyan.dev` points to this VPS's IP address in your DNS settings.*
```bash
sudo certbot --nginx -d lalithadithyan.dev
```

## 3. Install the C2 Server
Upload the `c2_server` folder to your VPS home directory (e.g., `/home/ubuntu/c2_server`).
```bash
cd ~/c2_server
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Start the Flask Server in the background using Gunicorn (Runs on internal port 5000):
```bash
gunicorn --workers 1 --bind 127.0.0.1:5000 server:app --daemon
```

## 4. Install Chisel (For the SSH over HTTPS Tunnel)
```bash
curl https://i.jpillora.com/chisel! | bash
```
Start the Chisel server in the background (Runs on internal port 8080):
```bash
chisel server --port 8080 --reverse &
```

## 5. Configure Nginx to Route Everything over HTTPS (Port 443)
Edit the Nginx configuration file:
`sudo nano /etc/nginx/sites-available/default`

Replace the `location /` block inside the port 443 server block with the following:

```nginx
server {
    server_name lalithadithyan.dev;
    
    # ... (Leave the SSL configurations certbot added) ...

    # 1. Route the hidden SSH Tunnel requests to Chisel
    location /ssh {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
    }

    # 2. Route everything else (Dashboard UI) to Flask
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        client_max_body_size 50M; # allow big image uploads
    }
}
```
Test and restart Nginx:
```bash
sudo nginx -t
sudo systemctl restart nginx
```

---

## 6. How to connect from the Linux GPU Machine

**To establish the Reverse SSH Tunnel:**
Download `chisel` on the Linux GPU machine, then run:
```bash
chisel client wss://lalithadithyan.dev/ssh R:2222:localhost:22
```
*(This perfectly masks your SSH tunnel as encrypted HTTPS traffic, bypassing your firewall!).*

**To log in from your Laptop:**
```bash
ssh -p 2222 your_gpu_user@lalithadithyan.dev
```
