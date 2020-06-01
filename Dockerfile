FROM tensorflow/tensorflow:2.1.0-py3

LABEL maintainer="gaborpelesz@gmail.com"

RUN apt-get update \
    && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    yasm \
    pkg-config \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavformat-dev \
    libpq-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install numpy

WORKDIR /

ENV OPENCV_VERSION="4.2.0"
RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
    && unzip ${OPENCV_VERSION}.zip \
    && mkdir /opencv-${OPENCV_VERSION}/cmake_binary \
    && cd /opencv-${OPENCV_VERSION}/cmake_binary \
    && cmake -DBUILD_TIFF=ON \
    -DBUILD_opencv_java=OFF \
    -DWITH_CUDA=OFF \
    -DWITH_OPENGL=ON \
    -DWITH_OPENCL=ON \
    -DWITH_IPP=ON \
    -DWITH_TBB=ON \
    -DWITH_EIGEN=ON \
    -DWITH_V4L=ON \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
    -DPYTHON_EXECUTABLE=$(which python3.7) \
    -DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -DPYTHON_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
    .. \
    && make install \
    && rm /${OPENCV_VERSION}.zip \
    && rm -r /opencv-${OPENCV_VERSION}

RUN apt-get update && apt-get install tesseract-ocr -y

ENV FLASK_ENV=production

COPY ./app /app

RUN pip3 install --upgrade pip
RUN pip3 install --trusted-host pypi.python.org -r /app/requirements.txt

# installing craft-text-detector downgrades opencv to 3.4.8.29
# did not find any compatibility issues with 4.2.0.34 so upgrading
RUN pip3 install -U opencv-python==4.2.0.34

CMD [ "python3", "/app/app.py" ]