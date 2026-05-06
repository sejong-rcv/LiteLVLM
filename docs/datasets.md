# Prepare Datasets

We provide instructions for preparing the dataset for LiteLVLM.

## Download Datasets
For Referring Expression Segmentation, we adopt the COCO Referring Expression Comprehension datasets: RefCOCO, RefCOCO+, and RefCOCOg.

Download links and directory structure:
- `images: COCO2014` - [train2014](http://images.cocodataset.org/zips/train2014.zip)
- [RefCOCO](https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip)
- [RefCOCO+](https://web.archive.org/web/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip)
- [RefCOCOg](https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip)

Download the data and orgnaize as follows:

```
data
├── Refer_Segm
│   ├── refcoco
│   │   ├── instances.json 
│   │   ├── refs(google).p
│   │   ├── refs(unc).p
│   ├── refcoco+
│   ├── refcocog
│   ├── coco_2014
│   │   ├── train2014
│   │   │   ├── COCO_train2014_000000000009.jpg
│   │   │   ├── COCO_train2014_000000000025.jpg

```
