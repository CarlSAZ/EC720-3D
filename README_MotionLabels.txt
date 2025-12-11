Note on motion label data:
- Currently aligned with the RGB timestamp (probably should be aligned to depth instead...)
- I.e. "D:\Bronn\rgbd_bonn_crowd\rgb\1548339819.87426.png" <=> "D:\Bronn\rgbd_bonn_crowd\motionLabels\1548339819.87426.png"
Label values:
- 255 = very high confidence of dynamic object
- 150 - ambiguous whether depth noise or dynamic object (ignore these points)
- 50 - depth pixels were invalid (ignore)
- 0 - very high confidence of static object

Bonn Data source:
https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/index.html

Step 1:
- Download the subsampled ground truth point cloud (https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/rgbd_bonn_groundtruth_1mm_section.zip)
- Download the rbgd_bonn_crowd data files (https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/rgbd_bonn_crowd.zip)

Step 2:
Extract to somewhere on local computer
i.e.
path/bonn/
-> rgbd_bonn_groundtruth.ply
-> rgbd_bonn_crowd/
	-> depth/
	-> rgb
	-> depth.txt
	-> groundtruth.txt
Step 3:
Grab the motion label frames from the project drive:
https://drive.google.com/drive/folders/1lwJ406PGVaBSG5NFcvY-xaQ7xu3nmbaG?usp=drive_link
- Unzip them to the same folder as the rgb and depth data for the dataset (i.e. path/bonn/rgbd_bonn_crowd/motionLabels)

Step 4:
Download the 
- make the truth table 
- Or download the BonnTruth mat file (much larger and slower download) from Google drive
