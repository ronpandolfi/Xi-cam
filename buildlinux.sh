#!/usr/bin/env bash
sudo rm -rf build
python buildexe.py build
tar -zcvf dist/Xi-cam_Linux-x86_64-2.7
