#!/usr/bin/env bash
source ~/virtualenv/python2.7_with_system_site_packages/bin/activate #works on travis
source venv/bin/activate # works on local
sudo rm -rf build
export PYTHONPATH=$PYTHONPATH:/usr/lib/python2.7/dist-packages/
python buildexe.py build
tar -zcvf Xi-cam.linux-x86_64-2.7 build/exe.linux-x86_64-2.7
