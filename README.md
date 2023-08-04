# Dozer_Push_project

# Derivation
The volume of between two survey surfaces is determined by integrating over the surfaces of each unit square in the delta surface ans summing all of these values. The surface of each square in the rectilinear grid was determined using bilinear interpolation. The delta surface itself is broken into squares along the easting and northing plane (x-y plane). The formula that represents bilinear interpolation on the unit square grid is given by:

$$f(x,y) \approx  \left(\begin{array}{cc} 
1-x & x\end{array}\right) \left(\begin{array}{cc} 
f(0,0) & f(0,1)\\
f(1,0) & f(1,1)
\end{array}\right) \left(\begin{array}{cc} 
1-y\\
y
\end{array}\right)$$

This formula represents the surface between the unit square: $(0,0), (0,1), (1,0), (1,1)$. We know the values of $f(0,0), f(0,1), f(1,0), f(1,1)$ to be $\Delta z_{(0,0)}, \Delta z_{(0,1)}, \Delta z_{(1,0)}, \Delta z_{(1,1)}$ which are the elevations of the delta surface. For sake of simplifying nomenclature, let $\Delta z = z$. Expanding the equation and simplifying the formula we can get somthing simpler to integrate over than a matrix:

$$f(x,y) \approx (1-y)[(1-x)z_{(0,0)}+xz_{(1,0)}]+y[(1-x)z_{(0,1)}+xz_{(1,1)}]$$

Now, evaluate the following equation:

$$\int \int f(x,y) dx\,dy\$$
$$\int \int (1-y)[(1-x)z_{(0,0)}+xz_{(1,0)}]+y[(1-x)z_{(0,1)}+xz_{(1,1)}] dx\,dy\$$
$$\frac{1}{4} z_{(0,0)} x^2 y^2 - \frac{1}{2} z_{(0,0)} x^2 y - \frac{1}{2} z_{(0,0)} x y^2 + z_{(0,0)}xy - \frac{1}{4} z_{(1,0)} x^2 y^2 + \frac{1}{2} z_{(1,0)} x^2 y - \frac{1}{4} z_{(0,1)} x^2 y^2 + \frac{1}{2} z_{(0,1)} x y^2 +c_{1}x + c_{2} + \frac{1}{4} z_{(1,1)} x^2 y^2$$

Over the region of interest:

$$\int_{y=0}^1 \int_{x=0}^1 f(x,y) dx\,dy\ = \int_{y=0}^1 \int_{x=0}^1 (1-y)[(1-x)z_{(0,0)}+xz_{(1,0)}]+y[(1-x)z_{(0,1)}+xz_{(1,1)}] dx\,dy\$$
$$\int_{y=0}^1 \int_{x=0}^1 f(x,y) dx\,dy\ = \frac{1}{4} (z_{(0,0)} + z_{(0,1)} + z_{(1,0)} + z_{(1,1)})$$

Now, $\frac{1}{4} (z_{(0,0)} + z_{(0,1)} + z_{(1,0)} + z_{(1,1)})$ is the mean of the delta elevations on each corner of the unit square. Therefore, to determine the total volume moved over the 15 minute period, we find the delta surface for a 1m by 1m grid take the mean of the four delta elevations and do this for each unque square along the easting-northing plane. 


#DBSCAN
DBSCAN (Density-Based Spatial CLustering of Applications with Noise) is an machine learning technique used to identify clusters of data.
https://medium.com/@tarammullin/dbscan-parameter-estimation-ff8330e3a3bd
https://medium.com/@tarammullin/dbscan-2788cfce9389

<!--
z00 - (z00)/2 + (z10)/2 - (z00)/2 + (z01)/2 + (z00)/4 - (z10)/4 - (z01)/4 + (z11)/4

(z00)/4 + (z10)/4 + (z01)/4 + (z11)/4

mean(z00 + z01 + z10 + z11)

