# Notes book.

in this repository, I'm going to upload my personal notes, codes, This is a form of study for me and a way of quick reference.

## Requirements

* [Linux Ubuntu 16.04]
* [Python 2/3]

## Install Pack "ROS, Gazebo, Turtlebot"

### ROS
Firt install ROS Follow the steps on the page:
* http://wiki.ros.org/kinetic/Installation/Ubuntu

### Gazebo.

  `curl -ssL http://get.gazebosim.org | sh`

  `sudo apt-get install libsdformat4`

  `sudo apt-get install libgazebo7`

  `sudo apt-get install ros-kinetic-gazebo-ros`

  `sudo apt-get install ros-kinetic-gazebo-ros-pkgs ros-kinetic-gazebo-ros-control`

### Turtlebot

  `sudo apt-get install ros-kinetic-turtlebot`

  `sudo apt-get install ros-kinetic-turtlebot-apps`

  `sudo apt-get install ros-kinetic-turtlebot-interactions`

  `sudo apt-get install ros-kinetic-turtlebot-simulator`

  `sudo apt-get install ros-kinetic-kobuki-ftdi`

  `sudo apt-get install ros-kinetic-ar-track-alvar-msgs`

  `sudo apt-get install python-rosdep python-wstool ros-kinetic-ros`

  `rosdep update`

or ............

follow this video, if you
https://www.youtube.com/watch?v=36O6OGOJG1E

## Lauch ROS,Gazebo and Turtlebot .

First launch
  `roscore`
In different tabs

  Gazebo simulation:
  `roslaunch turtlebot_gazebo turtlebot_world.launch`

  Rviz "view sensor"
  `roslaunch turtlebot_rviz_launchers view_robot.launch`

  Move robot
  `roslaunch turtlebot_teleop keyboard_teleop.launch`

## Notas de compilación.

  Tutorials ROS:
    http://wiki.ros.org/ROS/Tutorials

  Para ver las rutas que se estan ejecutnado.
  `printenv | grep ROS`

  para crear catkin workspace
  `mkdir -p ~/catkin_ws/src
   cd ~/catkin_ws/
   catkin_make`

   Para ejecutar
   `source devel/setup.bash`

## Notas uso de ROS


  `rosnode info`
  Se puede ver la informacion del nodo.

  `rostopic -h`
  despliega un menu de ayuda!

  `rostopic list -v`
  muestra los topicos con cartacteristicas.

  `rostopic echo [topic]`
  Muestra los datos que se estan escribiendo

  `rostopic type [topic]`
  muestra el tipo de mensaje que se esta publicando

    `rosmsg show [Resulta rostopic type]`
    muestra como es el detalle del mesaje usado.


## Config KinectV2

for use Kinect Sensor firt install this package:
  https://github.com/OpenKinect/libfreenect2#debianubuntu-1404

  https://openkinect.org/wiki/Main_Page

For use with ROS install:
  https://github.com/code-iai/iai_kinect2#install

My Kinect use this config:
    device serial: 001038264247
    device firmware: 4.0.3911.0

### Use KinectV2

Set PATH
  souurce devel/setup.bash

  roslaunch kinect2_bridge kinect2_bridge.launch reg_metod:=cpu depth_method:=cpu publish_tf:=true fps_limit:=1

  `roslaunch kinect2_bridge kinect2_bridge.launch publish_tf:=true fps_limit:=10`

  `rosrun kinect2_viewer kinect2_viewer kinect2 sd cloud`


## Install additional

  https://github.com/OctoMap/octomap/wiki/Compilation-and-Installation-of-OctoMap


 Pandas, Install pip

 scatter para graficar.






## Author

* **JuanD Valenciano** - (juan-da3@hotmail.com)


## License

This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <http://unlicense.org>
