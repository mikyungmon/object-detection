import argparse
import os
import time

import numpy as np

import torch
import torch.utils.data
import torch.utils.tensorboard
import tqdm
from yolo_v3_anchor_box import YOLOv3

import datasets
import utils


def evaluate(model, path, iou_thres, conf_thres, nms_thres, image_size, batch_size, num_workers, device):
    # 모델을 evaluation mode로 설정
    model.eval()

    # 데이터셋, 데이터로더 설정
    dataset = datasets.ListDataset(path, image_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             collate_fn=dataset.collate_fn)

    labels = []
    sample_metrics = []  # List[Tuple] -> [(TP, confs, pred)]
    entire_time = 0
    for _, images, targets in tqdm.tqdm(dataloader, desc='Evaluate method', leave=False):
        if targets is None:
            continue

        # Extract labels
        labels.extend(targets[:, 1].tolist())

        # Rescale targets
        targets[:, 2:] = utils.xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= image_size

        # Predict objects
        start_time = time.time()
        with torch.no_grad():
            images = images.to(device)
            outputs = model(images)
            outputs = utils.NMS(outputs, conf_thres, nms_thres)
        entire_time += time.time() - start_time

        # Compute true positives, predicted scores and predicted labels per batch
        sample_metrics.extend(utils.get_batch_statistics(outputs, targets, iou_thres))

    # Concatenate sample statistics
    if len(sample_metrics) == 0:
        true_positives, pred_scores, pred_labels = np.array([]), np.array([]), np.array([])
    else:
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]

    # Compute AP
    precision, recall, AP, f1, ap_class = utils.ap_per_class(true_positives, pred_scores, pred_labels, labels)

    # Compute inference time and fps
    inference_time = entire_time / dataset.__len__()
    fps = 1 / inference_time

    # Export inference time to miliseconds
    inference_time *= 1000

    return precision, recall, AP, f1, ap_class, inference_time, fps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=3, help="number of epoch")
    parser.add_argument("--gradient_accumulation", type=int, default=1, help="number of gradient accums before step")
    parser.add_argument("--multiscale_training", type=bool, default=True, help="allow for multi-scale training")
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--num_workers", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--data_config", type=str, default="C:/Users/mkflo/object-detection/files/YOLO/config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, default='weights/darknet53.conv.74',
                        help="if specified starts from checkpoint model")
    parser.add_argument("--image_size", type=int, default=416, help="size of each image dimension")
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    now = time.strftime('%y%m%d_%H%M%S', time.localtime(time.time()))

    # Tensorboard writer 객체 생성
    log_dir = os.path.join('logs', now)
    os.makedirs(log_dir, exist_ok=True)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)

    # 데이터셋 설정값을 가져오기
    data_config = utils.parse_data_config(args.data_config)
    train_path = data_config['train']
    valid_path = data_config['valid']
    num_classes = int(data_config['classes'])
    class_names = utils.load_classes(data_config['names'])

    # 모델
    model = YOLOv3(args.image_size, num_classes).to(device)
    # model = YOLOv3().to(device)

    # 데이터셋, 데이터로더 설정
    dataset = datasets.ListDataset(train_path, args.image_size, augment=True, multiscale=args.multiscale_training)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.num_workers,
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # optimizer 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # learning rate scheduler 설정
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    # 현재 배치 손실값을 출력하는 tqdm 설정
    loss_log = tqdm.tqdm(total=0, position=2, bar_format='{desc}', leave=False)

    # Train code.
    for epoch in tqdm.tqdm(range(args.epoch), desc='Epoch'):
        # 모델을 train mode로 설정
        model.train()

        # 1 epoch의 각 배치에서 처리하는 코드
        for batch_idx, (_, images, targets) in enumerate(tqdm.tqdm(dataloader, desc='Batch', leave=False)):
            step = len(dataloader) * epoch + batch_idx

            # 이미지와 정답 정보를 장치(CPU or GPU)로 복사
            images = images.to(device)
            targets = targets.to(device)

            # 순전파 (forward), 역전파 (backward)
            loss, outputs = model(images, targets)
            loss.backward()

            # 기울기 누적 (Accumulate gradient)
            if step % args.gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()

            print('======================================')
            print('Loss: {:.6f}'.format(loss.item()))

            for i, yolo_layer in enumerate(model.yolo_layers):
                print('--------------------------------------')
                print('loss_bbox_{}'.format(i + 1), yolo_layer.metrics['loss_bbox'], step)
                print('loss_conf_{}'.format(i + 1), yolo_layer.metrics['loss_conf'], step)
                print('loss_cls_{}'.format(i + 1), yolo_layer.metrics['loss_cls'], step)
                print('loss_layer_{}'.format(i + 1), yolo_layer.metrics['loss_layer'], step)

            print('\ntotal_loss: ', loss.item(), step, '\n')
            print('======================================')

            # 총 손실값 출력
            # loss_log.set_description_str('Loss: {:.6f}'.format(loss.item()))
            #
            # # Tensorboard에 훈련 과정 기록
            # tensorboard_log = []
            # for i, yolo_layer in enumerate(model.yolo_layers):
            #     writer.add_scalar('loss_bbox_{}'.format(i + 1), yolo_layer.metrics['loss_bbox'], step)
            #     writer.add_scalar('loss_conf_{}'.format(i + 1), yolo_layer.metrics['loss_conf'], step)
            #     writer.add_scalar('loss_cls_{}'.format(i + 1), yolo_layer.metrics['loss_cls'], step)
            #     writer.add_scalar('loss_layer_{}'.format(i + 1), yolo_layer.metrics['loss_layer'], step)
            # writer.add_scalar('total_loss', loss.item(), step)

        # lr scheduler의 step을 진행
        scheduler.step()

        # 검증 데이터셋으로 모델을 평가
        precision, recall, AP, f1, _, _, _ = evaluate(model,
                                                      path=valid_path,
                                                      iou_thres=0.5,
                                                      conf_thres=0.5,
                                                      nms_thres=0.5,
                                                      image_size=args.image_size,
                                                      batch_size=args.batch_size,
                                                      num_workers=args.num_workers,
                                                      device=device)

        print('val_precision', precision.mean(), epoch)
        print('val_recall', recall.mean(), epoch)
        print('val_mAP', AP.mean(), epoch)
        print('val_f1', f1.mean(), epoch)

        # # 검증 데이터셋으로 모델을 평가
        # precision, recall, AP, f1, _, _, _ = evaluate(model,
        #                                               path=valid_path,
        #                                               iou_thres=0.5,
        #                                               conf_thres=0.5,
        #                                               nms_thres=0.5,
        #                                               image_size=args.image_size,
        #                                               batch_size=args.batch_size,
        #                                               num_workers=args.num_workers,
        #                                               device=device)
        #
        # # Tensorboard에 평가 결과 기록
        # writer.add_scalar('val_precision', precision.mean(), epoch)
        # writer.add_scalar('val_recall', recall.mean(), epoch)
        # writer.add_scalar('val_mAP', AP.mean(), epoch)
        # writer.add_scalar('val_f1', f1.mean(), epoch)

        # checkpoint file 저장
        save_dir = os.path.join('checkpoints', now)
        os.makedirs(save_dir, exist_ok=True)
        dataset_name = os.path.split(args.data_config)[-1].split('.')[0]
        torch.save(model.state_dict(), os.path.join(save_dir, 'yolov3_{}_{}.pth'.format(dataset_name, epoch)))