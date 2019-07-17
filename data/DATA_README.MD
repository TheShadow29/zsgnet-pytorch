# Dataset Preparation
Here I outline the steps to prepare the following datasets:
- Flickr30k Entities
- ReferIt
- Flickr30k new splits
- Visual Genome splits

We convert the annotations for each dataset into `.csv` file with the format:
img_name,x1,y1,x2,y2,query(ies)

For training files multiple queries are present in the same line, for validation, test we have only one query per line.

The project directory is $ROOT

### Flickr30k Entities
Current directory is located at $FLICKR=/some_path/flickr30k
1. To get the Flickr30k Images you need to fill a form whose instructions can be found here http://shannon.cs.illinois.edu/DenotationGraph/. Un-tar the file and save it under $FLICKR/flickr30k_images
1. `git clone https://github.com/BryanPlummer/flickr30k_entities.git`. Unzip the annotations. At this point the directory should look like:
```
$FLICKR
|-- flickr30k_entities
    |-- Annotations
    |-- Sentences
    |-- test.txt
    |-- train.txt
    |-- val.txt
|-- flickr30k_images
|-- results.json
```
1. Make a symbolic link to $FLICKR here using `ln -s $FLICKR $ROOT/data/flickr30k`
1. Now we prepare the flickr30k entities dataset using
```
cd $ROOT
python data/prepare_flickr30k.py
```
1. The above code does the following:
   + Convert the annotations in `.xml` to a single `.json` file (because it is easier to deal with dictionaries and better to read only once). It is saved in $FLICKR/all_ann.json
   + Create train/val/test `.csv` files under $FLICKR/csvs/flickr_normal/{train/val/test}.csv
1. At this point the directory structure should look like this:
```
$FLICKR
|-- all_ann_2.json
|-- all_annot_new.json
|-- csvs
    |-- flickr_normal
	|-- train.csv
	|-- val.csv
	|-- test.csv
|-- flickr30k_entities
    |-- Annotations
    |-- Sentences
    |-- test.txt
    |-- train.txt
    |-- val.txt
|-- flickr30k_images
```

### ReferIt (Refclef)
Current directory is located at $REF=/some_path/referit
1. Follow the download links at https://github.com/lichengunc/refer to setup referit (refclef). Your folder structure after downloading the images (image subset of imageclef) and the annotations should look like this:
```
$REF
|-- images
    |-- saiapr_tc12_images
|-- refclef
    |-- instances.json
    |-- refs(berkeley).p
    |-- refs(unc).p
```
1. We use only the `berkeley` split to be consistent with previous works.
1. Now we again convert to csv format. First create a symbolic link, and then run `prepare_referit.py`
```
cd $ROOT/data
ln -s $REF referit
cd $ROOT
ptyhon data/prepare_referit.py
```
