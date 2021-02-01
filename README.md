# Team-Qualifier-Scoring-System  

The system used to calculate the conversion score of each players in OFCT.

## Prerequisites

This project is known to compile on `g++ 10.2.0` under `Ubuntu 20.04`.

## Get, Install, Compile, Build and Run

1. Make sure you have `git` installed.
```sh
sudo apt install git
```
2. Get the [latest stable release](http://ceres-solver.org/installation.html#getting-the-source-code) of `Ceres`, and follow the instructions below, which can also be found [here](http://ceres-solver.org/installation.html#linux).
```sh
sudo apt-get install cmake libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev libsuitesparse-dev
```
3. Untar the tar, and build `Ceres`.
```sh
tar zxf ceres-solver-2.0.0.tar.gz
mkdir ceres-bin
cd ceres-bin
cmake ../ceres-solver-2.0.0
make -j3 # you can change the number depending on the number of cores you have.
make test
make install
```
4. Clone the repository and build the Team Qualifier Scoring System.
```sh
git clone https://github.com/OFCT-Devs/Team-Qualifier-Scoring-System.git
cd Team-Qualifier-Scoring-System
mkdir build
cd build
cmake ..
make
./ScoringSystem
```
