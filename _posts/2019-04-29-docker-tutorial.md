---
layout: post
title: Docker Tutorial
date: 2019-04-29
categories: DevOps
description: Docker 原理、命令、部署介绍
---

<!--START figure-->
<div class="figure">
  <a href="https://ws4.sinaimg.cn/large/006tNc79ly1g2knue62lxj30fy0bkmxc.jpg" data-lightbox="docker_icon">
    <img src="https://ws4.sinaimg.cn/large/006tNc79ly1g2knue62lxj30fy0bkmxc.jpg" width="70%" alt="docker_icon" referrerPolicy="no-referrer"/>
  </a>
</div>
<!--END figure-->

<br>

本文主要是学习和使用 Docker 过程中的一些简单的笔记。

<br>

### 目录

1. [安装](#安装)
2. [Docker 基本概念](#docker-基本概念)
3. [Docker 常用指令](#docker-常用指令)
4. [Docker 部署简单的例子](#docker-部署简单的例子)

---

### 安装

- MacOS

  方法 1：使用 Homebrew，运行以下命令，安装完打开 Docker.app 即可在命令行使用 Docker

  ```bash
  brew cask install docker
  ```

  方法 2：从 [这里](https://docs.docker.com/docker-for-mac/install/) 安装 Docker Hub，安装完打开 Docker.app 即可在命令行使用 Docker

- Ubuntu

  使用脚本安装（自动下载脚本并安装 Docker 及其依赖包）

  ```bash
  wget -qO- https://get.docker.com/ | sh
  ```

  安装后会有以下提示

  ```bash
   If you would like to use Docker as a non-root user, you should now consider
   adding your user to the "docker" group with something like:

   sudo usermod -aG docker $USER

   Remember that you will have to log out and back in for this to take effect!  
  ```

  非 root 没有权限 access docker engine，如果要以非 root 身份运行 Docker，需要执行以下命令将当前用户添加到 docker group 中。

  ```bash
  sudo usermod -aG docker $USER
  ```

  安装完成后需要开启 docker 服务

  ```bash
  sudo service docker start
  # or
  sudo systemctl start docker

  # 开启自启动 docker 服务
  sudo systemctl enable /usr/lib/systemd/system/docker.service
  ```

<br>

### Docker 基本概念

Docker 是一种虚拟化技术，其出现以及流行是为了解决 **开发环境和线上生产环境不一致** 的问题，降低系统部署开发和部署的成本，将运行环境也纳入到版本管理中。传统的虚拟机技术是去模拟一个完整的操作系统，因此通常占用更多的系统资源，冗余步骤多、启动虚度慢，而 Docker 本质上是利用命名空间对进程、网络和文件系统进行隔离，更加轻量级，所需的资源更少。

Docker 最基本的两个概念是：镜像（image）、容器（container）。

<br>

#### image（镜像）

image 可以理解为一层一层的**只读层**堆叠在一起的统一视角，除了最底层之外每层可读层都有指针指向下一层。Docker 统一文件系统（the union file system）将不同的层整合成一个文件系统，这样对用于隐藏了多层的存在，在用户角度只存在一个文件系统。

<!--START figure-->
<div class="figure">
  <a href="https://ws2.sinaimg.cn/large/006tNc79ly1g2jkvftmg2j30vu0acmzg.jpg" data-lightbox="docker_image">
    <img src="https://ws2.sinaimg.cn/large/006tNc79ly1g2jkvftmg2j30vu0acmzg.jpg" width="80%" alt="docker_image" referrerPolicy="no-referrer"/>
  </a>
</div>
<!--END figure-->

<br>

#### image layer（镜像层）

image 是由一层一层的 image layer 组成的，一个 image layer 包含了几样东西：

  1. layer id
  2. layer metadata
  3. 指向 parent layer 的指针（如果位于最底层，则没有该指针）
  4. 该层对文件系统的改变

<!--START figure-->
<div class="figure">
  <a href="https://ws2.sinaimg.cn/large/006tNc79ly1g2km5zmd07j30iy03v0sz.jpg" data-lightbox="docker_iamge_layer">
    <img src="https://ws2.sinaimg.cn/large/006tNc79ly1g2km5zmd07j30iy03v0sz.jpg" width="80%" alt="docker_iamge_layer" referrerPolicy="no-referrer"/>
  </a>
</div>
<!--END figure-->

<br>

#### container（容器）

container 和 image 的定义几乎一样，也是多层的统一视角，区别在于 container 的**顶层是可读写的**。即 container = image + R/W layer，如下图所示。容器运行时进程对文件的读写操作都作用于该读写层。

<!--START figure-->
<div class="figure">
  <a href="https://ws2.sinaimg.cn/large/006tNc79ly1g2jl05dzq6j30zy09uju9.jpg" data-lightbox="docker_container">
    <img src="https://ws2.sinaimg.cn/large/006tNc79ly1g2jl05dzq6j30zy09uju9.jpg" width="80%" alt="docker_container" referrerPolicy="no-referrer"/>
  </a>
</div>
<!--END figure-->

#### running container（运行态容器）

运行态的 container 定义为一个静态的 container 加上一个隔离的进程空间（以及运行于其中的进程）和隔离的文件系统。

<!--START figure-->
<div class="figure">
  <a href="https://ws3.sinaimg.cn/large/006tNc79ly1g2jmb2s3hej30zw0cu0wn.jpg" data-lightbox="docker_running_container">
    <img src="https://ws3.sinaimg.cn/large/006tNc79ly1g2jmb2s3hej30zw0cu0wn.jpg" width="80%" alt="docker_running_container" referrerPolicy="no-referrer"/>
  </a>
</div>
<!--END figure-->

<br>

### Docker 常用指令

- 容器的创建和启动

    ```bash
    # 创建一个容器，做的事情是在 image 上面添加一个可读写层
    docker create <image-id>
    # 启动容器，为容器创建一个隔离的进程空间
    docker start <container-id>

    # run 指令相当于前两条指令的组合
    docker run <image-id>
    ```

- 查看镜像/容器

  ```bash
  # 列出所有（顶层）镜像。加上 -a 参数可以将底层的可读层也列出来
  docker images [-a]
  # 列出所有（运行态）容器。加上 -a 参数可以将非运行态的容器也列出来
  docker ps [-a]
  ```

- 进入一个运行中的容器

  ```bash
  # 进入容器打开 bash
  docker exec -it <container-id> bash
  ```

- 查看运行中容器的 stdout

  ```bash
  docker logs <container-id>
  ```

- 查看镜像/容器顶层元数据

  ```bash
  docker inspect <container-id> or <image-id>
  ```

- 递归查看镜像顶层往下的可读层

  ```bash
  docker history <image-id>
  ```

- 停止容器运行

  ```bash
  # stop 向容器内进程发送 SIGTERM 信号，可以被捕捉
  docker stop <container-id>
  # kill 向容器内进程发送 SIGKILL 信号，不能被捕捉，强行退出
  docker kill <container-id>  
  ```


- 删除镜像/容器

  ```bash
  # 删除容器的可读写层
  docker rm <container-id>  
  # 删除镜像最顶层的可读层
  docker rmi <iamge-id>  
  ```

- 容器 -> 镜像（将容器顶层的读写层转化为不可变的可读层）

  ```bash
  docker commit <container-id>
  ```

- **build**

  ```bash
  docker build
  ```

  build 指令会根据 Dockerfile 里面的 FROM 指令获取基本镜像，然后重复执行：

  1. run（镜像 -> 运行态容器）
  2. 修改（对可读写层进行修改）
  3. commit （将可读写层转化为只读层，构成新的镜像）


- 镜像部署

  ```bash
  # save 创建一个镜像的压缩文件
  docker save <image-id>
  # load 载入镜像的压缩文件
  docker load

  # export 创建一个 tar 文件，并且移除了元数据和不必要的层，将多个层整合成了一个层。
  # tar 文件再 import 到 Docker 中后，通过 docker images –tree 命令只能看到一个镜像； save 后的镜像则不同，它能够看到这个镜像的历史镜像
  docker export <container-id>
  ```

<br>

### Docker 部署简单的例子

1. 生成镜像压缩包

   ```bash
   # 查看当前镜像
   docker images
   ```
   <!--START figure-->
   <div class="figure">
     <a href="https://ws3.sinaimg.cn/large/006tNc79ly1g2kiydcsdhj30yi04iadp.jpg" data-lightbox="show_curr_image">
       <img src="https://ws3.sinaimg.cn/large/006tNc79ly1g2kiydcsdhj30yi04iadp.jpg" width="100%" alt="show_curr_image" referrerPolicy="no-referrer"/>
     </a>
   </div>
   <!--END figure-->

   ```bash
   # 默认将镜像文件打印到标准输出，使用 gzip 生成压缩包
   docker save mypyenv | gzip > mypyenv-latest.tar.gz
   ```
   <!--START figure-->
   <div class="figure">
     <a href="https://ws3.sinaimg.cn/large/006tNc79ly1g2kizaj0epj30yk02udi1.jpg" data-lightbox="zip_image">
       <img src="https://ws3.sinaimg.cn/large/006tNc79ly1g2kizaj0epj30yk02udi1.jpg" width="100%" alt="zip_image" referrerPolicy="no-referrer"/>
     </a>
   </div>
   <!--END figure-->

2. 加载镜像压缩包

   ```bash
   # load 加载镜像，默认从标准输入加载， -i 参数制定从归档文件加载
   docker load -i mypyenv-latest.tar.gz
   ```
   <!--START figure-->
   <div class="figure">
     <a href="https://ws2.sinaimg.cn/large/006tNc79ly1g2kj185keij30p007qq92.jpg" data-lightbox="load_image">
       <img src="https://ws2.sinaimg.cn/large/006tNc79ly1g2kj185keij30p007qq92.jpg" width="100%" alt="load_image" referrerPolicy="no-referrer"/>
     </a>
   </div>
   <!--END figure-->

   ```bash
   # 查看当前镜像
   docker images
   ```
   <!--START figure-->
   <div class="figure">
     <a href="https://ws4.sinaimg.cn/large/006tNc79ly1g2kj0x16jqj30yg04i77v.jpg" data-lightbox="show_curr_image2">
       <img src="https://ws4.sinaimg.cn/large/006tNc79ly1g2kj0x16jqj30yg04i77v.jpg" width="100%" alt="show_curr_image2" referrerPolicy="no-referrer"/>
     </a>
   </div>
   <!--END figure-->

<br>

### 参考文章

- [Docker Documentation](https://docs.docker.com/)
- [Visualizing Docker Containers and Images](http://merrigrove.blogspot.sg/2015/10/visualizing-docker-containers-and-images.html)
- [Docker Tutorial](https://juejin.im/entry/5b19e350e51d45069f5e1d66)

<br><br>
