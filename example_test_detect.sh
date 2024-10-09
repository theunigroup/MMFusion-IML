source data.sh
exp='./experiments/ec_example_phase2.yaml'
ckpt='./ckpt/late_fusion_detection.pth'

$pint test_detection.py --gpu $gpu --exp $exp --ckpt $ckpt --manip $columbia_manip --auth $columbia_auth
$pint test_detection.py --gpu $gpu --exp $exp --ckpt $ckpt --manip $cover_manip --auth $cover_auth
$pint test_detection.py --gpu $gpu --exp $exp --ckpt $ckpt --manip $dso1_manip --auth $dso1_auth
$pint test_detection.py --gpu $gpu --exp $exp --ckpt $ckpt --manip $cocoglide_manip --auth $cocoglide_auth
$pint test_detection.py --gpu $gpu --exp $exp --ckpt $ckpt --manip $casiav1_manip --auth $casiav1_auth