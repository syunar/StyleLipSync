pip install gdown
gdown --id 16sLXo9eR3de1xJxFpM79bdIL86ebeLIL
gdown --id 1VJkK56Qtx5bBOOgHmG4vfxK2XvfCleWy
gdown --id 16xLfZ-OgNbu0aqWsAmkG785Hv7Xy3uEH

mv AlexandriaOcasioCortez_0_3k.pth ckpts
mv w_avg.pth ckpts
mv AlexandriaOcasioCortez_0.zip data
unzip data/AlexandriaOcasioCortez_0 -d data/
rm -rf data/__MACOSX