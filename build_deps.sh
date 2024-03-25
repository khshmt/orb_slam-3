echo "Configuring and building Thirdparty/DBoW2 ..."

cd Thirdparty/DBoW2
if [ -d build ]; then
    rm -rf build
fi
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

cd ../../g2o

echo "Configuring and building Thirdparty/g2o ..."
if [ -d build ]; then
    rm -rf build
fi
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

cd ../../Sophus

echo "Configuring and building Thirdparty/Sophus ..."
if [ -d build ]; then
    rm -rf build
fi
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

cd ../../../

echo "Uncompress vocabulary ..."

cd Vocabulary
tar -xf ORBvoc.txt.tar.gz
cd ..
