#!/usr/bin/env bash
sudo rm -rf build
# export PYTHONPATH=$PYTHONPATH:/usr/lib/python2.7/dist-packages/
echo $PYTHONPATH
echo "/usr/lib/python2.7/dist-packages/" > ~/virtualenv/python2.7_with_system_site_packages/local/lib/python2.7/site-packages/global.pth
python -c "import numpy; print numpy.__file__"
python buildexe.py build
tar -zcvf Xi-cam.linux-x86_64-2.7 build/exe.linux-x86_64-2.7
