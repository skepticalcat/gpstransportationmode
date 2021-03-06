#### LSTM based transport mode detection with geolife [1] dataset and pytorch

Achieves approx. 75-80% point based accuracy for car, walk, bus, train and subway modes.
70% of trajectories are at least 80% correct.

Written out of personal interest

#### How to run:

* Run raw_data_loader.py with path to geolife dataset as argument (raw_data_loader.py /path/to/geolife/data/)
* Afterwards run classifier.py
* Please note that data creation (label extraction and enrichment will take a while)

##### References

[1] Yu Zheng, Lizhu Zhang, Xing Xie, Wei-Ying Ma. Mining interesting locations and travel sequences from GPS trajectories. In Proceedings of International conference on World Wild Web (WWW 2009), Madrid Spain. ACM Press: 791-800.
