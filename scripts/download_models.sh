# download t5 model
cd model/engine
git lfs install
git clone https://huggingface.co/google/flan-t5-base
cd ../..

# download motion/shape mean & std
cd data
gdown 1-4mm9KCuEWhIurD0k4XDVcN0Dmh5tQAf
unzip smpl.zip
rm smpl.zip

gdown 1OMzn-MORlrtWgS-ODGzM5DcZrCQ08zje
unzip shape_data.zip
rm shape_data.zip

mv data/* .
rm -r data
cd ..

# download model weight
gdown 1RtQLmYosluLfqCk8XOnB0cx0KDV1EZKp
unzip pretrain_model.zip
rm pretrain_model.zip

# download smpl model for visualization
echo -e "\nYou need to register at https://smpl.is.tue.mpg.de"
read -p "Username (SMPL):" username
read -p "Password (SMPL):" password
wget --post-data "username=$username&password=$password" \
    'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip&resume=1' \
    -O './data/SMPL_python_v.1.1.0.zip' --no-check-certificate --continue

mkdir -p data/smpl
unzip ./data/SMPL_python_v.1.1.0.zip -d data/
rm ./data/SMPL_python_v.1.1.0.zip
mv ./data/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl ./data/smpl/SMPL_NEUTRAL.pkl
rm -rf ./data/SMPL_python_v.1.1.0
