# Pruned DFT spread FBMC

Pruned DFT spread FBMC is a novel modulation scheme with the remarkable properties of a low PAPR, low latency transmissions and a high spectral efficiency.
It is closely related to FBMC, OFDM and SC-FDMA and I first proposed it in my [PhD thesis](http://publik.tuwien.ac.at/files/publik_265168.pdf), see Chapter 6. 
A more detailed description can be found in R. Nissel and M. Rupp, [“Pruned DFT Spread FBMC: Low PAPR, Low Latency, High Spectral Efficiency”](https://ieeexplore.ieee.org/document/8360161/), IEEE Transactions on Communications, 2018. 


The Python script simulates a pruned DFT spread FBMC transmission over a doubly-selective channel (time-variant multipath propagation) and compares the performance to OFDM, SC-FDMA and FBMC.


Furthermore, the included classes (QAM, DoublySelectiveChannel, OFDM, FBMC) can also be reused in future projects.


* A [Matlab code](https://github.com/rnissel/Pruned-DFT-s-FBMC_Matlab) of pruned DFT spread FBMC with much more features can also be found on GitHub.


## Usage

Just run **Simulation.py** in Python 3. 

Requires the packages: numpy, scipy(sparse), matplotlib, time and mpl_toolkits.mplot3d.  



## Simulation Results* 
\* for "nr_rep = 1000"

### Pruned DFT spread FBMC has the same PAPR as SC-FDMA:

![](png/Figure_5.png)

----------
### Pruned DFT spread FBMC outperforms SC-FDMA in doubly-selective channels:

![](png/Figure_2.png)

Note that pruned DFT spread FBMC does not require a CP and thus has a higher data rate than conventional SC-FDMA.

----------
### Pruned DFT spread FBMC has superior spectral properties, comparable to FBMC: 

![](png/Figure_4.png)

----------
### Pruned DFT spread FBMC dramatically reduces the ramp-up and ramp-down period of FBMC:
![](png/Figure_3.png)



## Please Cite Our Paper

    @ARTICLE{Nissel2018,
		author  = {R. Nissel and M. Rupp},
		journal = {IEEE Transactions on Communications},
		title   = {Pruned {DFT} Spread {FBMC}: Low {PAPR},Low Latency, High Spectral Efficiency},
		year    = {2018},
		volume  = {},
		number  = {},
		pages   = {}, 
		doi     = {10.1109/TCOMM.2018.2837130},
		ISSN    = {},
		month   = {},
	}


## References
- R. Nissel and M. Rupp, [“Pruned DFT Spread FBMC: Low PAPR, Low Latency, High Spectral Efficiency”](https://ieeexplore.ieee.org/document/8360161/), IEEE Transactions on Communications, 2018 to appear.
- R. Nissel, [“Filter bank multicarrier modulation for future wireless systems”](http://publik.tuwien.ac.at/files/publik_265168.pdf), Dissertation, TU Wien, 2017.

