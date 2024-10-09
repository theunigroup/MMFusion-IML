source data.sh
exp="./experiments/ec_example.yaml"
ckpt="./ckpt/late_fusion_localization.pth"
res="./results/ec_example/localization"
mkdir -p $res
$pint test_localization.py --gpu $gpu --exp $exp --ckpt $ckpt --manip $columbia_manip > $res/Columbia.txt
clear
$pint test_localization.py --gpu $gpu --exp $exp --ckpt $ckpt --manip $cover_manip > $res/COVER.txt
clear
$pint test_localization.py --gpu $gpu --exp $exp --ckpt $ckpt --manip $dso1_manip > $res/DSO-1.txt
clear
$pint test_localization.py --gpu $gpu --exp $exp --ckpt $ckpt --manip $cocoglide_manip > $res/CocoGlide.txt
clear
$pint test_localization.py --gpu $gpu --exp $exp --ckpt $ckpt --manip $casiav1_manip > $res/Casiav1.txt