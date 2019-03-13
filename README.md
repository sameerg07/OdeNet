# XNORNet

## Folder Structure
!! - Don't touch.
╒═════════════════════════════════╤═════════════════╤══════════════════════════════════════════════════════════════╕
│ File/Folder Name                │ Action          │ Comments                                                     │
╞═════════════════════════════════╪═════════════════╪══════════════════════════════════════════════════════════════╡
│ binary_layers.py                │ !!              │ Dont Touch - It's the predecessor of XNORNet                 │
│                                 │                 │   BinaryNet. Has the layers of BinaryNet                     │
├─────────────────────────────────┼─────────────────┼──────────────────────────────────────────────────────────────┤
│ binary_ops.py                   │ !!              │ Dont Touch - Contains operations of BinaryNet                │
├─────────────────────────────────┼─────────────────┼──────────────────────────────────────────────────────────────┤
│ faces95                         │ Download/Create │ Download faces95 dataset and place it in your                │
│                                 │                 │  project folder                                              │
├─────────────────────────────────┼─────────────────┼──────────────────────────────────────────────────────────────┤
│ faces95_cnn_relu.py             │ Edit/Run        │ Edit (if needed) and Run this with Python.                   │
│                                 │                 │  Runs CNN on Faces95 with Relu activation                    │
├─────────────────────────────────┼─────────────────┼──────────────────────────────────────────────────────────────┤
│ faces95_cnn_selu.py             │ Edit/Run        │ Edit (if needed) and Run this with Python.                   │
│                                 │                 │  Runs CNN on Faces95 with Selu activation                    │
├─────────────────────────────────┼─────────────────┼──────────────────────────────────────────────────────────────┤
│ faces95dataWithImgSalMBD.pickle │ Create          │ Create by running file in                                    │
│                                 │                 │  './pyimgsaliency/pyimgsaliency/load_faces95_data_imgSal.py' │
├─────────────────────────────────┼─────────────────┼──────────────────────────────────────────────────────────────┤
│ faces96                         │ Download/Create │ Download faces96 dataset and place it in                     │
│                                 │                 │  your project folder                                         │
├─────────────────────────────────┼─────────────────┼──────────────────────────────────────────────────────────────┤
│ faces96_cnn_relu.py             │ Edit/Run        │ Edit (if needed) and Run this with Python.                   │
│                                 │                 │  Runs CNN on Faces96 with Relu activation                    │
├─────────────────────────────────┼─────────────────┼──────────────────────────────────────────────────────────────┤
│ faces96_cnn_selu.py             │ Edit/Run        │ Edit (if needed) and Run this with Python.                   │
│                                 │                 │  Runs CNN on Faces96 with Selu activation                    │
├─────────────────────────────────┼─────────────────┼──────────────────────────────────────────────────────────────┤
│ faces96dataWithImgSalMBD.pickle │ Create          │ Create by running file in                                    │
│                                 │                 │  './pyimgsaliency/pyimgsaliency/load_faces96_data_imgSal.py' │
├─────────────────────────────────┼─────────────────┼──────────────────────────────────────────────────────────────┤
│ grimace                         │ Download/Create │ Download grimace dataset and place it in                     │
│                                 │                 │  your project folder                                         │
├─────────────────────────────────┼─────────────────┼──────────────────────────────────────────────────────────────┤
│ grimace_cnn_relu.py             │ Edit/Run        │ Edit (if needed) and Run this with Python.                   │
│                                 │                 │  Runs CNN on Grimace with Relu activation                    │
├─────────────────────────────────┼─────────────────┼──────────────────────────────────────────────────────────────┤
│ grimace_cnn_selu.py             │ Edit/Run        │ Edit (if needed) and Run this with Python.                   │
│                                 │                 │  Runs CNN on Grimace with Selu activation                    │
├─────────────────────────────────┼─────────────────┼──────────────────────────────────────────────────────────────┤
│ grimacedataWithImgSalMBD.pickle │ Create          │ Create by running file in                                    │
│                                 │                 │  './pyimgsaliency/pyimgsaliency/load_grimace_data.py'        │
├─────────────────────────────────┼─────────────────┼──────────────────────────────────────────────────────────────┤
│ load_grimace_data.py            │ Edit/Run        │ Run or Edit, if needed,                                      │
│                                 │                 │  to generate 'grimaceData.pickle' file. Check                │
│                                 │                 │ 'datasetFolderName' variable inside the file is correct.     │
│                                 │                 │  It must point to the directory                              │
│                                 │                 │  having Grimace dataset.                                     │
├─────────────────────────────────┼─────────────────┼──────────────────────────────────────────────────────────────┤
│ orl                             │ Download/Create │ Download orl dataset and place it in                         │
│                                 │                 │  your project folder                                         │
├─────────────────────────────────┼─────────────────┼──────────────────────────────────────────────────────────────┤
│ outputs                         │ Create          │ Make this directory manually                                 │
├─────────────────────────────────┼─────────────────┼──────────────────────────────────────────────────────────────┤
│ pyimgsaliency                   │ !!;Edit/Run     │ Edit only the 'load_' prefixed files present                 │
│                                 │                 │  in the 'pyimgsaliency' subfolder.                           │
│                                 │                 │  Don't touch rest                                            │
├─────────────────────────────────┼─────────────────┼──────────────────────────────────────────────────────────────┤
│ README.md                       │ !!              │ You're reading this, want to mess with it?                   │
├─────────────────────────────────┼─────────────────┼──────────────────────────────────────────────────────────────┤
│ rename_files.py                 │ Edit/Run        │ Run this if you need to rename images, with                  │
│                                 │                 │  a number prefix to indicate a class, of some                │
│                                 │                 │  dataset. Use if necessary.                                  │
├─────────────────────────────────┼─────────────────┼──────────────────────────────────────────────────────────────┤
│ resize-script.sh                │ Edit/Run        │ Run this if you need to resize images of some                │
│                                 │                 │  dataset. Use if necessary.                                  │
├─────────────────────────────────┼─────────────────┼──────────────────────────────────────────────────────────────┤
│ run.sh                          │ Edit/Run        │ Use this if needed to automate training. Check script        │
│                                 │                 │  inside, it's real easy to follow.                           │
├─────────────────────────────────┼─────────────────┼──────────────────────────────────────────────────────────────┤
│ xnor_layers.py                  │ !!              │ Dont Touch - Contains code of XNORNet layers.                │
╘═════════════════════════════════╧═════════════════╧══════════════════════════════════════════════════════════════╛

## Things to check before running:
1. FIRST TIME ONLY : Make sure you've downloaded and placed the dataset as needed.
2. FIRST TIME ONLY : Now edit resize-script.sh : change `originalGrimace` to the dataset folder name. Eg: If my dataset name if `faces95` then I'll replace every occurence of `originalGrimace` with `faces95`. Then first time do : `chmod +x resize-script.sh`. For first and further runs do : 						`./resize-script.sh`.
3. FIRST TIME ONLY : Now edit rename_files.py. Follow same changes as in step 2.
4. FIRST TIME ONLY : Then generate pickle files by using `load_` methods. For Faces95 and Faces96. they are found inside the `pyimgsaliency` directory. For grimace it's found inside and outside. In the methods, make sure to change the `datasetFolderName` variable to have the same name as the dataset you're interested in generating the pickle for. 

> Note: If you use the `load_` methods for Faces95 and Faces96 in `pyimgsaliency` then it'll generate pickle files with MBD applied. To generate without MBD, you've to do the following : 
	1. Create a copy of load_grimace_data.py
	2. Change `datasetFolderName` variable to point to Faces95/Faces96 directory.

> Note: In every `load_` file, at end you can find a `with open(...) as xxx` line, you can change the pickle file name by changing the string inside `...`.

5. Then, in the `_cnn_` files you want to run, make sure that the path in `pickledDatset` variable points to the `.pickle` file you generated. Then you can run any of the commands as shown below.

> Note: If you make any modifications to "`load_`" methods, you've to redo from step 4. If you make changes to image/ wish to do something then redo from step 1.

## To Run:
`
python3 faces95_cnn_selu.py
python3 faces96_cnn_selu.py
python3 grimace_cnn_selu.py
python3 grimace_cnn_relu.py
python3 faces96_cnn_relu.py
python3 faces95_cnn_relu.py
`

## Reference
* Rastegari et al. [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://arxiv.org/abs/1603.05279)
