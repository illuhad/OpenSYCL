#!/bin/bash 
set -o xtrace

echo '[hipsycl]' >> /etc/pacman.conf
echo "Server = http://repo.urz.uni-heidelberg.de/sycl${1}/archlinux/x86_64" >> /etc/pacman.conf

pacman-key --init
wget -q -O - http://repo.urz.uni-heidelberg.de/sycl/hipsycl.asc | pacman-key --add -
pacman-key --lsign-key E967BA09716F870320089583E68CC4B9B2B75080
pacman -Syu


