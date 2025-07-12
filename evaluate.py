import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
from train import *
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [0, 0, 0]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 255, 255]

    return mask_rgb


def img_writer(inp):
    (mask,  mask_id, rgb) = inp
    if rgb:
        mask_name_tif = mask_id + '.png'
        mask_tif = label2rgb(mask)
        cv2.imwrite(mask_name_tif, mask_tif)
    else:
        mask_png = mask.astype(np.uint8)
        mask_name_png = mask_id + '.png'
        cv2.imwrite(mask_name_png, mask_png)


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to  config")
    arg("-o", "--output_path", type=Path, help="Path where to save resulting masks.", required=True)
    arg("-t", "--tta", help="Test time augmentation.", default=None, choices=[None, "d4", "lr"])
    arg("--rgb", help="whether output rgb images", action='store_true')
    return parser.parse_args()


def main():
    seed_everything(42)
    args = get_args()
    config = py2cfg(args.config_path)
    args.output_path.mkdir(exist_ok=True, parents=True)
    model = Supervision_Train.load_from_checkpoint(os.path.join(config.weights_path, config.test_weights_name+'.ckpt'), config=config)
    model.cuda()
    evaluator = Evaluator(num_class=config.num_classes)
    evaluator.reset()
    model.eval()
    if args.tta == "lr":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip()
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                tta.Rotate90(angles=[90,180,270]),
                tta.Scale(scales=[0.5,0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False)
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)

    test_dataset = config.test_dataset

    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=4,
            pin_memory=False,
            drop_last=False,
        )
        results = []
        for input in tqdm(test_loader):
            # raw_prediction NxCxHxW
            raw_predictions = model(input[0].cuda())

            image_ids = input[2]
            masks_true = input[1]

            raw_predictions = nn.Softmax(dim=1)(raw_predictions)
            predictions = raw_predictions.argmax(dim=1)

            for i in range(raw_predictions.shape[0]):
                mask = predictions[i].cpu().numpy()
                evaluator.add_batch(pre_image=mask, gt_image=masks_true[i].cpu().numpy())
                
                mask[mask>0] = 255
                mask = mask.astype(np.uint8)
                masks_true = masks_true.cpu().numpy()[0]
                #print(np.unique(masks_true))
                masks_true[masks_true >0] == 255
                masks_true = masks_true.astype(np.uint8)
                
                save_dir = os.path.join(args.output_path,str(image_ids[0])+"_res"+".jpg")
                #print(save_dir)
                cv2.imwrite(save_dir,mask)
                #cv2.imwrite("./fig_results/unetformer/"+str(image_ids[0])+"_label"+".jpg",masks_true)
    iou_per_class = evaluator.Intersection_over_Union()
    f1_per_class = evaluator.F1()
    dsc_per_class = evaluator.DSC()
    sen_per_class = evaluator.Sen()
    spe_per_class = evaluator.Spe()
    
    print(sen_per_class,spe_per_class)
    
    OA = evaluator.OA()
    for class_name, class_iou, class_f1 in zip(config.classes, iou_per_class, f1_per_class):
        print('F1_{}:{}, IOU_{}:{}'.format(class_name, class_f1, class_name, class_iou))
    
    print('mIOU:{}, OA:{}'.format(np.nanmean(iou_per_class), OA))
    print('DSC:{}, SEN:{}, SPE:{}'.format(np.nanmean(dsc_per_class), sen_per_class[0], spe_per_class[0]))

    print("& "+ str(np.round(np.nanmean(iou_per_class)*100,2)) \
            + " & " + str(np.round(np.nanmean(dsc_per_class)*100,2)) \
            + " & " + str(np.round(OA*100,2)) \
            + " & " + str(np.round(sen_per_class[0]*100,2)) \
            + " & " + str(np.round(spe_per_class[0]*100,2))
            )

    if 0:
        t0 = time.time()
        mpp.Pool(processes=mp.cpu_count()).map(img_writer, results)
        t1 = time.time()
        img_write_time = t1 - t0
        print('images writing spends: {} s'.format(img_write_time))


if __name__ == "__main__":
    main()
