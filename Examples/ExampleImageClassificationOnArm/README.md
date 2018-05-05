# Getting Started

This project demonstrates running Tensorflow on arm32 to run image classification using a MobileNet model trained on flowers.  The Tensorflow model is relatively small (~5 MB).  The project can be built and run using Docker.  Docker supports ARM emulation via [qemu](https://www.qemu.org/) out of the box.

Build the project for x64:
```
%> docker build --rm -f Dockerfile-x64 -t exampleimageclassificationonarm:x64 .
```

Run the project for x64:
```
%> docker run --rm exampleimageclassificationonarm:x64
Classification prediction for images/iris-flower.jpg: iris with probability 1.00
Classification prediction for images/rose-flower.jpg: roses with probability 0.98
Classification prediction for images/tulip-flower.jpg: tulips with probability 1.00
```

Build the project for arm32 on your x64 device:
```
%> docker build --rm -f Dockerfile-arm32 -t exampleimageclassificationonarm:arm32 .
```

Run the project for arm32 on your x64 device.  This will run significantly slower than the x64 version, and may take a while to start up on first run:
```
%> docker run --rm exampleimageclassificationonarm:arm32
qemu: Unsupported syscall: 389
Classification prediction for images/iris-flower.jpg: iris with probability 1.00
Classification prediction for images/rose-flower.jpg: roses with probability 0.98
Classification prediction for images/tulip-flower.jpg: tulips with probability 1.00
```
