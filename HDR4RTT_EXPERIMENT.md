# FOCER-v2 HDR4RTT same-source experiment

This directory is an isolated copy of FOCER-v2. The network, frozen DETR
teacher, fixed stops 0 through 14, and top-3 object-query exposure routing are
unchanged. Only the experimental protocol is new.

The immutable split contains 1,264 training images, 168 validation images,
150 untouched test images, and 14 temporal guard frames. Video sources use
100-original-frame blocks. Frames within three original indices of a split
boundary are assigned to the guard set. The still-image source is grouped by
camera and calendar day. Every paper category (person, bicycle, car, truck)
is present in train, validation, and test.

Training has two phases. First, train on 1,264 images for at most 20 epochs and
select the epoch with the highest validation pooled four-category COCO mAP.
Second, start from a fresh router initialization and refit on the combined
1,432 train+validation images for exactly the selected number of epochs. Only
then evaluate the 150-image test set. The paper's fixed 15-exposure method is
evaluated on exactly the same test annotations and detector checkpoint.

The test split must not be used for architecture selection. If the final test
delta is disappointing, create a new experiment before inspecting test-driven
changes rather than repeatedly tuning on these 150 images.
