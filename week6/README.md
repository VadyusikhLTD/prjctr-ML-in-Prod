# Drift detection algorithms

My data are tables that contains GPS data of public transport or vehicle location information (VLI).
Each tabel contain records for 1 day.


## Univariate drift detection
For univariate drift detection I've choose number of new records 
in 10 minute intervals. I've use two methods - Kolmogorov-Smirnov test
and bounds base method.

They located in file `univariate_update_frequency.py` functions 
`detect_ks_drift` and `detect_bounded_drift` respectfully.

Visualisation is below. You also can create plot - use code
in `univariate_update_frequency.py` file. 

To calculate referral distribution you can use function `calculate_reference_distribution`
its' inputs is `table_distribution_list` (historical distributions list to calculate
referral distribution on ) and `solving_func` is the function that process input distributions
and output the referral one, by default it is `np.median`. 

**Referral VLI updates distributions**
![Kolmogorov-Smirnov test visualisation](docs/images/Referral_VLI_updates_distribution.jpg)


**Kolmogorov-Smirnov test visualisation**
![Kolmogorov-Smirnov test visualisation](docs/images/Kolmogorov-Smirnov_drift_test.jpg)

**Bounds base method visualisation**
![Bounds base method visualisation](docs/images/bounds_drift_detection.jpg)


## Multivariate drift detection
For multivariate drift detection I've choosed coordinates. 
I've use Kolmogorov-Smirnov test for data drift detection.

It located in file `multivariate_drift_detection.py` function `detect2d_ks_drift`.

Visualisation is below. You also can create plot - use code
in `multivariate_drift_detection.py` file. 

To calculate referral distribution you can use function `calculate_reference_distribution2d`
its' inputs is `table_distribution_list` (historical distributions list to calculate
referral distribution on ) and `solving_func` is the function that process input distributions
and output the referral one, by default it is `np.median`. 

**Referral VLI updates distributions**

![Coordinate reference distribution visualisation](docs/images/coordinate_reference_distribution_visualisation.jpg)


**Kolmogorov-Smirnov test visualisation**

![Coordinate diff distribution](docs/images/coordinate_diff_distribution.jpg)
