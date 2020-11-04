
CUR_DIR=$(pwd)

docker run --name gfootball --gpus all \
           --mount type=bind,source=${CUR_DIR},target=/kaggle-football/ \
           -d -it \
           -e DISPLAY=$DISPLAY \
           -v /tmp/.X11-unix:/tmp/.X11-unix:rw gfootball


