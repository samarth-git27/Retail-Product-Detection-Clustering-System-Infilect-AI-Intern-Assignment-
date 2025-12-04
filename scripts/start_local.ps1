# PowerShell script to create venv, install deps and run app
Write-Host "Creating virtualenv..."
python -m venv venv

Write-Host "Activating virtualenv..."
.\venv\Scripts\Activate.ps1

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Installing CPU PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

Write-Host "Installing requirements..."
pip install -r requirements.txt

Write-Host "Starting Flask app..."
python app.py
