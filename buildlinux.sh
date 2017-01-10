#!/usr/bin/env bash
source ~/virtualenv/python2.7_with_system_site_packages/bin/activate
source ~/venv/bin/activate
sudo rm -rf build
python buildexe.py build
tar -zcvf dist/Xi-cam_Linux-x86_64-2.7
