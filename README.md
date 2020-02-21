# GigaDepth
The goal of GigaDepth is to achieve similar results as HyperDepth but with CNNs. 

## Sensor
The used sensor is a structure core. With 2 global shutter IR cameras, 
its a good fit for this application. 
The pattern was extracted by pointing the sensor on a flat wall and fitting a plane. 
Not ideal since it relies on many (wrong) assumption:
* Plane not in focus
* Plane estimate is shit
* Our walls are not flat


## Dataset
The dataset used right now is mostly rendered in unity.
