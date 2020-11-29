
CUR_DIR=$(pwd)

docker run --name football-env --gpus all \
           --mount type=bind,source=${CUR_DIR},target=/kaggle-football/ \
           -d -it \
           -p 58888:8888 \
           -e DISPLAY=$DISPLAY \
           -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
           football-env         


