#! /bin/bash

# usage: acpp --help | ./cleanup_syclcchelp.sh
# output is in sylccout then and should be copied into using-acpp.md

sed "s/\[current value: .*\]/[current value: NOT SET]/g" > acppout

install_root=`cat acppout | grep -Po "Installation root: \K(.*)$"`
sed "s%$install_root%/install/path%g" -i acppout
sed -e "s%\(Plugin LLVM version: \)[[:digit:]][[:digit:]]*\(, can accelerate CPU: \)[[:alpha:]][[:alpha:]]*%\1<version>\2<bool>%g" -i acppout
sed -e "s%librt-backend-\w\w*.so%librt-backend-<name>.so%g" -i acppout
