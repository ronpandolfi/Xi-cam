#!/usr/bin/env bash
source ~/virtualenv/python2.7_with_system_site_packages/bin/activate
source venv/bin/activate
sudo rm -rf build
python buildexe.py build
tar -zcvf Xi-cam.linux-x86_64-2.7 build/exe.linux-x86_64-2.7
