# Label studio

To run [Label studio](https://github.com/heartexlabs/label-studio) from docker 

    docker run -p 8080:8080 -v `pwd`/mydata:/label-studio/data heartexlabs/label-studio:latest

App will start on http://0.0.0.0:8080, You can sing up and log in right there! 
