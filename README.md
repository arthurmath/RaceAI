python -m venv .venv
Set-ExecutionPolicy Unrestricted -Scope Process # ca autorise l'utilisator root 
.venv\Scripts\Activate.ps1
pip install tensorflow-directml-plugin
pip install numpy<2
pip install matplotlib pygame