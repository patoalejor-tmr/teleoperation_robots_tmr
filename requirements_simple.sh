git clone https://github.com/patoalejor-tmr/teleoperation_robots_tmr.git
cd teleoperation_robots_tmr
git submodule update --init --recursive

python3 -m venv .venv 
source .venv/bin/activate 

# Go to submodule #1 for retargeting 
cd third-party/retarget
git submodule update --init --recursive
pip install dex_retargeting
pip install -e ".[example]"

# Go to submodue #2 for IK
cd ..
cd third-party/pyroki
pip install -e .

# Install extra libraries
cd ..
pip install -r requirements.txt

# test
python source/show_vla_inference_2g_rby1.py