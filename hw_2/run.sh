cd code

rm -rf build 
mkdir build 
cd build 

cmake ..
make 

./main

cd ..
rm -rf build 
cd ..

