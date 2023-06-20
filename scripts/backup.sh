#/bin/sh
mkdir -p up/Docker up/cfg up/customs up/data up/deploy up/models up/scripts up/utils
cp *.py *.txt up/
cp -r Docker/* up/Docker/
cp -r cfg/* up/cfg/
cp -r customs/*.py up/customs/
cp -r data/*.yaml up/data/
cp -r deploy/* up/deploy/
cp -r models/*.py up/models/
cp -r scripts/*.sh up/scripts/
cp -r utils/*.py utils/wandb_logging up/utils/
 
