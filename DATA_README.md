# Dataset Loading

Note that the following steps uses annotations from the parent dataset converted into a fixed format (outlined below). For steps to reproduce the annotations see [DATA_PREP_README.md](./DATA_PREP_README.md)

The project directory is $ROOT

## Setup the directories
```
cd $ROOT/data
bash download_ann.sh
```

## Image Download

# Flickr30k Entities
Current directory is located at $FLICKR=/some_path/flickr30k

1. To get the Flickr30k Images you need to fill a form whose instructions can be found here http://shannon.cs.illinois.edu/DenotationGraph/. Un-tar the file and save it under $FLICKR/flickr30k_images
1. Make a symbolic link to the images using `ln -s $FLICKR/flickr30k_images $ROOT/data/flickr30k/flickr30k_images`

# ReferIt 
Current directory is located at $REF=/some_path/referit

1. Download the ImageClef subset for referit from https://github.com/lichengunc/refer. Download link: http://bvisionweb1.cs.unc.edu/licheng/referit/data/images/saiapr_tc-12.zip
1. Unzip to $REF/images/saiapr_tc12_images
1. Make a symbolic link to $REFER here using `ln -s $REF/images/saiapr_tc12_images $ROOT/data/referit/images/`

# Visual Genome
Current directory is located $VG=/some_path/visual_genome

1. See download page for Visual Genome (https://visualgenome.org/api/v0/api_home.html). Download the two image files to $VG/VG_100K and $VG/VG_100K_2
1. Make a symbolic link to $VG using 
```
ln -s $VG/VG_100K $ROOT/data/visual_genome/
ln -s $VG/VG_100K_2 $ROOT/data/visual_genome/
```

The remaining annotations are already existing, so should work out of the box. 

TODO:
- [ ] Create a script to automate the above given root directory (flickr30k still needs to be done manually).
